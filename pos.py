#! /usr/bin/env python3
from functools import partial
from itertools import groupby, filterfalse, tee, starmap, dropwhile
from collections import defaultdict
from typing import NamedTuple
from operator import is_, itemgetter
from segtok.segmenter import split_single as sentence_split
from segtok.tokenizer import word_tokenizer as word_split
from textblob_de.packages import pattern_de
from langdetect import detect
from pathlib import Path
import fileinput
import configparser
import uuid
import re
import os
import sys

# input:  german text
# output: line delimited sentences with each word annotated as
#         WORD/LEMMA/STTS_TAG
#         paragraph breaks indicated with an empty line

# For information on STTS tags see
# http://www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.htm

STTS = frozenset({
    'ADJA', 'ADJD', 'ADV', 'APPR', 'APPRART', 'APPO', 'APZR', 'ART', 'CARD', 'FM',
    'ITJ', 'KOUI', 'KOUS', 'KON', 'KOKOM', 'NN', 'NE', 'PDS', 'PDAT', 'PIS', 'PIAT',
    'PIDAT', 'PPER', 'PPOSS', 'PPOSAT', 'PRELS', 'PRELAT', 'PRF', 'PWS', 'PWAT',
    'PWAV', 'PAV', 'PTKZU', 'PTKNEG', 'PTKVZ', 'PTKANT', 'PTKA', 'TRUNC', 'VVFIN',
    'VVIMP', 'VVINF', 'VVIZU', 'VVPP', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN',
    'VMINF', 'VMPP', 'XY', '$,', '$.', '$('})


class Tag(NamedTuple):
    word:  str
    lemma: str
    pos:   str

    def __str__(self):
        return '%s|%s|%s' % self

    def __repr__(self):
        return 'Tag(%s|%s|%s)' % self


none = partial(is_, None)
fst = itemgetter(0)
snd = itemgetter(1)

def log(*message):
    print(*message, file=sys.stderr)

def truncate(s, maxchars):
    return (s[:maxchars] + '...') if len(s) > maxchars else s


# Config —————————————————————————————————————————————————————————————————————

class Config(NamedTuple):
    data_dir:     Path
    stanford_dir: Path
    parsed_dir:   Path

def _load_config():
    path = Path('~/.config/dzeug/dzeug.conf').expanduser()
    conf = configparser.ConfigParser()
    conf.read(str(path))
    try:
        data = Path(conf['dir']['data'])
        stanford = Path(conf['dir']['stanford'])
        parsed = Path(conf['dir']['parsed'])
        return Config(*map(Path.expanduser, (data, stanford, parsed)))
    except KeyError:
        sys.exit('''
Error: expected configuration file at {path} with data:\n
[dir]
data     = /path/to/dzeug_data_dir
stanford = /path/to/stanford_pos_tagger_dir
parsed   = /path/to/parsed_dir
''')

config = _load_config()


# Tag ————————————————————————————————————————————————————————————————————————

def new_german_postagger(stanford_dir, fast=False):
    from nltk.tag import StanfordPOSTagger
    modeltype = 'fast' if fast else 'hgc'
    modelfile = Path(stanford_dir,  'models', 'german-%s.tagger' % modeltype)
    jar = stanford_dir / 'stanford-postagger.jar'
    return StanfordPOSTagger(str(modelfile), str(jar))


def tokenize(text):
    # Yield successive word-split sentences.
    # Represent a new paragraph with a singleton list with a ¶ character
    # todo: escape rather than simple removal of pipe separator
    text = text.strip().replace('|', '')
    paragraphs = re.split(r'[ \t]*\n[ \t]*\n+', text)
    for para in paragraphs:
        for sent in sentence_split(para):
            yield word_split(sent)
        yield ['¶']


def _setup_tag():

    tagger = new_german_postagger(config.stanford_dir, fast=True)
    lemma_table = None

    def isparabreak(sent):
        # ignoring these from output
        return sent and sent[0] and sent[0][0] == '¶'

    def tagsent(sent):
        return [Tag(word, lemmatize(lemma_table, word, pos, n==0), pos)
                        for n, (word, pos) in enumerate(sent)]

    def make_lemma_table():
        # Create a lookup table of declensions
        path = config.data_dir / 'deutsch_lemma_unpacked.dat'
        lemmas = {}
        with open(path) as f:
            for form, lemma in map(str.split, f):
                lemmas[form] = lemma
                if lemma not in lemmas:
                    lemmas[lemma] = lemma
        return lemmas

    def _tag(text):
        nonlocal lemma_table
        if lemma_table is None:
            lemma_table = make_lemma_table()

        word_split_sentences = tokenize(text)
        tagged = tagger.tag_sents(word_split_sentences)
        groups = groupby(tagged, key=isparabreak)
        paragraphs = map(snd, filterfalse(fst, groups))
        for para in paragraphs:
            yield list(map(tagsent, para))

    return _tag

tag = _setup_tag()



# Lemmatization ——————————————————————————————————————————————————————————————

def lemma_adj(lemmas, word, begin_sent=False):
    # todo: handle present participle ending 'd', maybe don't lookup in lemmas
    if word in lemmas:
        return lemmas[word]
    # todo: Fix the superlative and/or comparative forms.
    superlatives = r'sten?$'
    word = re.sub(superlatives, '', word)
    lemma = pattern_de.predicative(word) # Adjektiv (stated 98% accuracy)
    # catch eg. Aachener, person from Aachen
    return lemma if word.islower() or begin_sent else lemma.capitalize()


def lemma_noun(lemmas, word):
    return lemmas.get(word, word)


def lemma_verb(word, pos):
    if pos == 'VVIZU':
        word = word.replace('zu', '')
    # todo: pattern_de gives weiß -> weeißen ??
    return pattern_de.lemma(word.lower())


def lemmatize(lemmas, word, pos, begin_sent=False):
    if pos.startswith('ADJ'):
        return lemma_adj(lemmas, word, begin_sent)
    if pos.startswith('N'):
        return lemma_noun(lemmas, word)
    if pos.startswith('V'):
        return lemma_verb(word, pos)
    return word.lower()



# Read ———————————————————————————————————————————————————————————————————————

def read_parsed(text):
    # Take an already parsed string or line-based iterator.
    # Return an iterator of paragraphs in a tree form:
    # document -> paragraphs -> sentences -> tags -> word|lemma|tag
    # todo: some check on input data

    def nonempty(s):
        return s.strip() != ''

    def maketags(line):
        # todo: assert the pos exists in a set of STTS tags
        return [Tag(*part.split('|', maxsplit=2)) for part in line.split()]

    lines = text.splitlines() if isinstance(text, str) else text
    # Create two groups. Each contains lists of paragraphs and
    # non-paragraphs (successive empty lines with no text)
    groups = groupby(lines, key=nonempty)
    # Remove the empty lines
    paragraphs = map(snd, filter(fst, groups))
    for para in paragraphs:
        # Yield a paragraph: a list of sentences with each sentence's element
        # converted to the Tag datatype
        yield list(map(maketags, para))


def reconstruct(sent):
    # Take a tagged sentence (a list of tags) and return a string representing
    # an approximation of what the sentence originally was with correct spacing
    # between tags. Likely better just to (optionally) return each sentence as
    # (sentence, parsed) pairs in the main tagging function than bother with this.

    def padded(seq, pad=None):
        yield from seq
        yield pad

    def pairwise(seq):
        a, b = tee(padded(seq))
        next(b, None)
        return zip(a, b)

    def combine(tag1, tag2):
        # forget about incorrect ” (U+201D right double quotation mark)
        space = tag1.word not in '„»([{‚›' and\
                        tag2 and tag2.word not in ';:.,!?“«…)]}‘‹'
        return tag1.word + ' ' if space else tag1.word

    def enclosing(match):
        delim, word, *_ = dropwhile(none, match.groups())
        return delim + word.strip() + delim

    # Compare each tag to the next and add a space if required according to
    # general punctuation rules.
    sent = ''.join(starmap(combine, pairwise(sent)))
    # Now replace enclosing delimiters which have the same open and close character.
    # just " for now
    sent = re.sub(r'(") ([^"]*)"', enclosing, sent)
    return sent



# Process ————————————————————————————————————————————————————————————————————

def output(tagged_paragraphs, file):
    for n, para in enumerate(tagged_paragraphs):
        if n > 0:
            print(file=file)
        for taglist in para:
            tagsent = ' '.join(map(str, taglist))
            print(tagsent, file=file)


def iter_documents(tagged, uuids):
    buffer = []
    for para in tagged:
        first_word = para[0][0][0]
        savename = uuids.get(first_word, None)
        if savename:
            yield buffer, savename
            buffer = []
        else:
            buffer.append(para)


def concat_documents(paths, lang_check):
    # one concatenated document performs much quicker than tagging individual docs
    # After each file insert a uuid as a dict key with value of the file's path
    uuids = {}
    docs = []
    for path in paths:
        uuid4 = uuid.uuid4().hex
        uuids[uuid4] = path
        with open(path) as f:
            text = f.read()
            if lang_check:
                assert_deutsch(text)
        docs.append(text)
        docs.append(uuid4)
    return '\n\n'.join(docs), uuids


def read_stdin(lang_check):
    text = ''.join(sys.stdin)
    if lang_check:
        assert_deutsch(text)
    return text


def read_input(paths, lang_check):
    if paths:
        return concat_documents(paths, lang_check)
    else:
        return read_stdin(lang_check), None


def output_multi(tagged, uuids, savedir=None):

    def empty_para():
        return [Tag('', '', '')]

    docs = iter_documents(tagged, uuids)

    for n, (doc, savename) in enumerate(docs):
        if savedir:
            path = savedir / savename
            with open(path, 'w') as f:
                output(doc, f)
            log('Saved', path)
        else:
            final = n == len(uuids)-1
            if not final:
                doc.append(empty_para())
            output(doc, sys.stdout)


def assert_deutsch(text):
    if detect(text) != 'de':
        message = truncate(text, maxchars=300) + '\n' +\
        'Is this German? Use option --no-lang-check to override.\nExiting...'
        sys.exit(message)



# Command line ———————————————————————————————————————————————————————————————

def commandline():
    from click import command, argument, option, Path as CPath

    @command()
    @argument('files', nargs=-1, type=CPath(exists=True))
    @option('--save', '-s', is_flag=True, help=(
        "For each named file provided, save its corresponding parsed document. "
        "The file is saved in the directory given by the config file's 'parsed' option. "
        "The filename is saved with an additional .dzeug extension. "
        "(Default: output is sent to stdout and input files are separated with two empty lines)"))
    @option('--no-lang-check', 'no_lang_check', is_flag=True,
        help='Skip language detection')
    def cli(files, save, no_lang_check):
        if save and not files:
            sys.exit('Error: the --save option applies only to named files, not stdin')

        paths = list(map(Path, files))
        text, uuids = read_input(paths, not no_lang_check)
        tagged = tag(text)

        if paths:
            savedir = config.parsed_dir if save else None
            output_multi(tagged, uuids, savedir)
        else:
            output(tagged, sys.stdout)
    cli()


if __name__ == '__main__':
    commandline()
