#! /usr/bin/env python3
from functools import partial
from itertools import (chain, starmap, islice, compress, groupby, repeat, tee)
from operator import itemgetter
import re
from dzeug.pos import read_parsed, tag, config
import os
import sys

flatten = chain.from_iterable
third = itemgetter(2)


def swap(pair):
    return pair[1], pair[0]


def chunkpad(seq, size, pad):
    it = chain(iter(seq), repeat(pad))
    return iter(lambda: tuple(islice(it, size)), (pad,) * size)


def valid_word(word):
    check = len(word) > 1 and not word.isupper()
    return check and re.match(r'^[A-Za-zÄÖÜäöüß-]+$', word)


def getvocab(doc, parts_of_speech, frequent_words):
    # This is "vocab" in a restricted sense. It ignores considers only nouns,
    # verbs and adjectives, and only those in the top 45,000 words.
    # The original document will likely have other lemmas, including compounds.

    def takeif(tag):
        return tag.lemma in frequent_words and\
             tag.pos.startswith(parts_of_speech) and\
             valid_word(tag.lemma)

    def no_word(tag):
        # simplify the tag to only the lemma and a part of speech,
        # using on the first letter of pos
        return (tag.lemma, tag.pos[0].lower())

    tags = filter(takeif, flatten(flatten(doc)))
    return set(map(no_word, tags))


def read_frequent(rank_min, rank_max):
    path = config.data_dir / 'deutsch_frequent_45000.dat'
    with open(path) as f:
        lines = map(str.rstrip, f)
        return list(islice(lines, rank_min, rank_max))


def show_stats(vocab):
    # More can go here...
    print('\nCount=%d' % (len(vocab)))


def show_vocab(vocab, columns=3, show_pos=True, show_rank=True):

    reset = "\x1b[0m"
    colorfmt  = "\x1b[38;5;%dm"
    gray = colorfmt % 239

    def format(word, pos, rank):
        s = '%6d %s %s' % (rank, pos, word)
        s = s.ljust(26)
        return gray + s[:9] + reset + s[9:]

    def rows_required(seq, column_len):
        rows = len(seq) // column_len
        if len(seq) % column_len != 0:
            rows += 1
        return rows

    words = list(starmap(format, sorted(vocab, key=third)))
    r = rows_required(words, columns)
    bycolumn = chunkpad(words, r, '')
    rows = zip(*bycolumn)

    print()
    for row in rows:
        print(''.join(row))


def show_simple_vocab(vocab):
    def spacejoin(t):
        return ' '.join(t)
    print('\n'.join(map(spacejoin, map(swap, vocab))))


def process(doc, parts_of_speech, minmax, simple):
    rank_min, rank_max = minmax
    freqlist = read_frequent(*minmax)

    # Slice according to the user-specified range, then make a
    # reversed word to index dict
    ranks = dict(map(swap, enumerate(freqlist, start=rank_min-1)))

    vocab = getvocab(doc, parts_of_speech, ranks)

    if simple:
        show_simple_vocab(vocab)
    else:
        vocab = list(map(lambda p: (p[0], p[1], ranks[p[0]]), vocab))
        show_vocab(vocab)
        show_stats(vocab)


def read_input(path):
    if path:
        with open(path) as f:
            return f.read()
    return sys.stdin.read()


# Command line ———————————————————————————————————————————————————————————————

def click_validate_range(minval, maxval):
    from click import BadParameter

    def f(ctx, param, value):
        if value is None:
            return (minval, maxval)

        pattern = r'^(?:(\d+),|,?(\d+)|(\d+),(\d+))$'
        m = re.match(pattern, value)
        if not m:
            raise BadParameter('\n\nRange must be positive number or comma-separated numbers')

        lo1, hi1, lo2, hi2 = m.groups()
        lo, hi = int(lo1 or lo2 or minval), int(hi1 or hi2 or maxval)
        if lo >= hi:
            raise BadParameter(f'\n\nThe low number must be less than the high number.')

        return max(lo, minval), min(hi, maxval)

    return f


def commandline():
    from click import command, argument, option, Path as CPath

    @command(help='')
    @argument('path', type=CPath(exists=True), required=False)
    @option('--nouns', '-n', is_flag=True,
        help="Include nouns")
    @option('--verbs', '-v', is_flag=True,
        help="Include verbs")
    @option('--adjectives', '-a', is_flag=True,
        help="Include adjectives")
    @option('--range', '-r', callback=click_validate_range(1, 45000),
        help='Limit vocab rank to this range [1,45000]')
    @option('--simple', '-s', is_flag=True,
        help="Show only lemmas as simple list, no formatting")
    @option('--tagged', '-t', is_flag=True,
        help="Input text is already tagged in dzeug format ")
    def cli(path, range, nouns, verbs, adjectives, simple, tagged):

        stts = [('NN',), ('V',), ('ADJA', 'ADJD')]
        pos = tuple(compress(stts, [nouns, verbs, adjectives]))
        pos = tuple(flatten(pos or stts))

        text = read_input(path)
        doc = read_parsed(text) if tagged else tag(text)
        process(doc, pos, range, simple)

    cli()


if __name__ == '__main__':
    commandline()
