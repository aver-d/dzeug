#! /usr/bin/env python3
from functools import partial
from itertools import islice, chain
from subprocess import run, Popen, PIPE
from collections import namedtuple
from dzeug.pos import Tag, read_parsed, reconstruct, config, snd, log, STTS
from click import command, argument, option, prompt, get_terminal_size, Path as CPath
from pathlib import Path
import re
import os
import sys
import textwrap
import jellyfish
import random

flatten = chain.from_iterable

reset     = "\x1b[0m"
underline = "\x1b[4m"
colorfmt  = "\x1b[38;5;%dm"
orange = colorfmt % 202
gray   = colorfmt % 248
markleft  = '\x00'
markright = '\x01'


Options = namedtuple('Options', 'use_pager show_titles shuffle format retry lemma_path')



# Formatting —————————————————————————————————————————————————————————————————

def shorten(text, n):
    # textwrap.shorten will strip whitespace at beginning (no option to prevent)
    # To work around this put dummy letter at beginning then replace
    space = text.startswith(' ')
    if space:
        text = 'A' + text[1:]
    text = textwrap.shorten(text, n, placeholder=' …')
    return ' ' + text[1:] if space else text

def shortenleft(text, n):
    # Shorten at beginning of string instead of the end
    tmp = shorten(rev(text), n)
    return rev(tmp)

def rev(text):
    return text[::-1]


def format_normal(wrap, sentence):
    result = '- ' + '\n  '.join(wrap(sentence))
    result = result.replace(markleft, orange).replace(markright, gray)
    return gray + result + reset


def format_centre(termwidth, sentence):

    s = sentence.find(markleft)
    e = sentence.find(markright)

    before, word, after = sentence[:s], sentence[s+1:e], sentence[e+1:]

    lenleft = termwidth // 3
    lenright = termwidth - lenleft - (e - s)

    before = shortenleft(before, lenleft).rjust(lenleft)
    after = shorten(after, lenright)

    return gray + before + orange + word + gray + after + reset


def as_title(path):
    suffix_part = ''.join(path.suffixes)
    no_ext = path.name[:-len(suffix_part)]
    return underline + no_ext.replace('_', ' ') + reset



# Search —————————————————————————————————————————————————————————————————————

def scan_paths(lemma, paths):
    # Return a list of paths including the lemma
    # Use grep -l (--files-with-matches) to do a quick scan before handing
    # over to python. If paths are given, restrict search to those.
    # Create a postings list if this becomes slow (not likely unless many, many docs)
    pattern = '\|%s\|' % lemma
    r = grep_r(paths)
    cmd = ('grep', '-lE'+r , '-m1', pattern, paths)
    out = run(cmd, stdout=PIPE)
    paths = map(Path, out.stdout.decode().splitlines())
    return list(paths)


def retrieve(path, lemma):

    def matchlemma(tag):
        return tag.lemma == lemma

    def haslemma(sent):
        return any(filter(matchlemma, sent))

    with open(path) as f:
        text = f.read()
        paragraphs = read_parsed(text)

    sentences = filter(haslemma, flatten(paragraphs))
    return sentences


def mark_selected(sentences, lemma):

    def mark(tag):
        if not tag.lemma == lemma:
            return tag
        marked_word = markleft + tag.word + markright
        return Tag(marked_word, tag.lemma, tag.pos)

    for sent in sentences:
        yield list(map(mark, sent))


def getsentences(lemma, paths, format, shuf=False, show_titles=True):
    paths = shuffle(paths) if shuf else\
                    sorted(paths)
    for path in paths:
        sents = retrieve(path, lemma)
        sents = mark_selected(sents, lemma)
        sents = map(reconstruct, sents)
        if show_titles:
            yield '\n' + as_title(path)
        yield '\n'.join(map(format, sents))


def getlemma(word, lemma_path):
    pattern = '^%s\t' % word
    cmd = ('grep', '-P', '-m1', pattern, lemma_path)
    out = run(cmd, stdout=PIPE)
    if out.returncode != 0:
        return word
    lemma = out.stdout.split()[1].decode()
    return lemma


def unique_lemmas(paths):

    def getword(tag):
        return tag[:tag.index(b'|')]

    no_punct_stts = filter(str.isalpha, STTS)
    stts_pattern = '(?:%s)' % '|'.join(no_punct_stts)
    pattern = '(?<=\|)[A-Za-zÄÖÜäöüß-]+\|' + stts_pattern

    r = grep_r(paths)
    cmd = ('grep', '-Poh'+r, pattern, paths)
    # Use Popen instead of run to allow streaming (maybe thousands/millions tags)
    # before taking only unique lemmas
    proc = Popen(cmd, stdout=PIPE)
    lines = iter(proc.stdout.readline, b'')
    unique = set(map(getword, set(lines)))
    return map(bytes.decode, unique)


def closest(word, lemmas):

    def compare(lemma):
        score = jellyfish.damerau_levenshtein_distance(word, lemma)
        return score, lemma

    def samefirst(lemma):
        # Consider only lemmas which match the first char. Different first char
        # is an unlikely mistake and confusing to present as alternative.
        # But include where only the case is different
        return word[0].lower() == lemma[0].lower()

    results = sorted(map(compare, filter(samefirst, lemmas)))
    top10 = map(snd, islice(results, 10))
    return list(top10)


def grep_r(paths):
    return 'r' if is_dir(paths) else ''



# Util ———————————————————————————————————————————————————————————————————————

def partition(predicate, seq):
    no, yes = [], []
    for item in seq:
        (no, yes)[predicate(item)].append(item)
    return no, yes


def shuffle(seq):
    random.shuffle(seq)
    return seq


def is_dir(path):
    return isinstance(path, Path) and path.is_dir()



# User interface —————————————————————————————————————————————————————————————

def ask_misspelling(word, choices):
    print('Alternatives...\n')
    for n, lemma in enumerate(choices, start=1):
        print(' %2d. %s' % (n, lemma))
    n = prompt('\nNumber 1-10', default='return to exit', type=int)
    valid = isinstance(n, int) and 1 <= n <= len(choices)
    return choices[n-1] if valid else None


def display_fail_paths(word, paths):

    def bullet(s):
        return f'  - {s}'

    maxpaths = 6
    names = [p.name for p in paths]
    n = len(names)
    adjunct = '' if n < maxpaths else\
                        '  and %d other provided files' % (n - maxpaths)
    pathlist = '\n'.join(map(bullet, names[:maxpaths]))+ '\n' + adjunct
    print(f'\n"{word}" does not appear in any of\n\n{pathlist}')


def display_fail_all(word, parsed_dir):
    print(f'"{word}" not found in any file in parsed directory: {parsed_dir}\n')


def display_sentences(text, use_pager=True):
    if not use_pager:
        print(text)
        return 0
    pager = os.environ.get('PAGER', 'less')
    cmd = run(pager, input=text.encode())
    return cmd.returncode



# Run ————————————————————————————————————————————————————————————————————————

def do(word, paths, options):

    o = options

    lemma = getlemma(word, o.lemma_path)
    valid_paths = scan_paths(lemma, paths)

    if not valid_paths:
        if not o.retry:
            # A not found message here…
            return 1

        if is_dir(paths):
            display_fail_all(word, paths)
            cache_check()
            choices = closest(word, cache_read())
        else:
            display_fail_paths(word, paths)
            choices = closest(word, unique_lemmas(paths))

        lemma = ask_misspelling(word, choices)
        if not lemma:
            return 1

        valid_paths = scan_paths(lemma, paths)

    result = getsentences(lemma, valid_paths, o.format, o.shuffle, o.show_titles)
    output = '\n'.join(result)
    if not output:
        log('No results')
        return 1
    return display_sentences(output, o.use_pager)



# Lemma cache ——————————————————————————————————————————————————————————————————————

def modtime(path):
    return path.stat().st_mtime


def cache_path():
    return config.parsed_dir / 'lemma_cache.dat'


def cache_check():
    files = config.parsed_dir.rglob('*.dzeug')
    if cache_stale(cache_path(), files):
        log('New documents found. Recaching lemmas...')
        cache_write(config.parsed_dir)


def cache_stale(path, files):
    t = modtime(path)
    def newer(p):
        return modtime(p) > t
    return any(map(newer, files))


def cache_write():
    lemmas = unique_lemmas(config.parsed_dir)
    with open(cache_path(), 'w') as f:
        f.write('\n'.join(lemmas))
        f.write('\n')


def cache_read():
    with open(cache_path()) as f:
        return list(map(str.rstrip, f))



# Command line ———————————————————————————————————————————————————————————————

def files_only(paths):
    nondirs, dirs = partition(Path.is_dir, map(Path, paths))
    files = set(nondirs)
    for dir in dirs:
        files.update(dir.rglob('*.dzeug'))
    return list(sorted(files))


def commandline():

    @command()
    @argument('word')
    @argument('paths', nargs=-1, type=CPath(exists=True))
    @option('--centre',   '-c', is_flag=True,
        help='Align word centrally when displaying')
    @option('--sentences','-s', is_flag=True,
        help="Show only sentences, no document titles")
    @option('--stdout',   '-t', is_flag=True,
        help='No pager, send directly to terminal')
    @option('--shuffle',  '-z', 'shuf', is_flag=True,
        help='Shuffle results order')
    @option('--no-retry', 'no_retry', is_flag=True,
        help='Exit immediately on not found')
    def cli(word, paths, centre, sentences, stdout, shuf, no_retry):

        # paths is either a list of paths or the parsed directory to search all
        # some kind of variant type would be useful
        paths = config.parsed_dir if not paths else files_only(paths)

        termwidth, _ = get_terminal_size()
        wrap = textwrap.TextWrapper(width=min(77, termwidth-1)).wrap

        if centre:
            format = partial(format_centre, termwidth)
        else:
            format = partial(format_normal, wrap)

        lemma_path = config.data_dir / 'deutsch_lemma_unpacked.dat'

        options = Options(not stdout, not sentences, shuf, format, not no_retry, lemma_path)

        code = do(word, paths, options)
        sys.exit(code)
    cli()


if __name__ == '__main__':
    commandline()
