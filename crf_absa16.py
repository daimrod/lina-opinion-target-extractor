#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from lxml import etree
from nltk.tokenize.casual import TweetTokenizer
import text_utils
import random
import math
from copy import deepcopy
import os.path
import codecs
from subprocess import Popen, PIPE
import re
import nltk
import tempfile
import treetaggerwrapper
import string


def extract_class_for_CRF(ipath, tokenizer=TweetTokenizer()):
    """Extract class information for all tokens #E/A.

    Args:
        ipath: The XML file to parse.
        tokenizer: The tokenizer to use.

    Returns:
        Returns a list of Sentence were dict consist of features
        to learn from by the CRF.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        tree = etree.parse(ifile)
    ret = []
    for review in tree.getroot().iterchildren():
        for sentences in review.iterchildren():
            for sentence in sentences.iterchildren():
                Sentence = {}
                text = ''
                opinions = []
                for element in sentence.iterchildren():
                    if element.tag == 'text':
                        text = element.text
                    if element.tag == 'Opinions':
                        for opinion in element.iterchildren():
                            opinions.append(opinion)
                if text is None:  # some texts might be missing, skip it
                    text = 'none'

                Sentence['raw_text'] = text
                Sentence['opinions'] = opinions
                Sentence['tokens'] = []
                Sentence['tokens'] = [token for token in tokenizer.tokenize(text) if token.strip() != '']
                Sentence['spans'] = text_utils.spans_retrieval(Sentence['raw_text'],
                                                               ' '.join(Sentence['tokens']))

                beg_class = False
                Sentence['tokens_class'] = []
                for (f1, t1) in Sentence['spans']:
                    w_class = 'O'
                    for opinion in opinions:
                        # EC: Are we sure we can extract from/to?
                        f2, t2 = opinion.get('from'), opinion.get('to')
                        if f2 is not None and t2 is not None:
                            f2, t2 = int(f2), int(t2)
                            if f1 >= f2 and t1 <= t2:
                                # Are we in the middle of a class segment?
                                if beg_class:
                                    prefix = 'I'
                                else:
                                    prefix = 'B'
                                    beg_class = True
                                w_class = prefix  # + '_' + opinion.get('category')
                                break
                            else:
                                beg_class = False
                    Sentence['tokens_class'].append(w_class)
                ret.append(Sentence)
    return ret


def write_class_for_CRF(Sentences, w_path, c_path, s_path):
    """Write the output of extract_class_for_CRF to be processed by Wapiti.

    $ paste w_path c_path > wapiti.crf;

    Args:
        Sentences: A list of Sentence as returned by extract_class_for_CRF.
        w_path: The file to write the words (surface form).
        c_path: The file to write the class.
        s_path: The file to write the spans.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(w_path, 'w', 'utf-8') as wfile:
        with codecs.open(c_path, 'w', 'utf-8') as cfile:
            with codecs.open(s_path, 'w', 'utf-8') as sfile:
                for Sentence in Sentences:
                    for (token, token_class, span) in zip(Sentence['tokens'],
                                                          Sentence['tokens_class'],
                                                          Sentence['spans']):
                        cfile.write(token_class + '\n')
                        wfile.write(token + '\n')
                        sfile.write(str(span[0]) + ' ' + str(span[1]) + '\n')

                    cfile.write('\n')
                    wfile.write('\n')
                    sfile.write('\n')


def write_tokenized_text(Sentences, tpath):
    """Write the tokenized text to tpath.

    Args:
        Sentences: A list of Sentence (see extract_class_for_CRF).
        tpath: The path to write the tokens.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(tpath, 'w', 'utf-8') as tfile:
        for Sentence in Sentences:
            tokens = []
            for token in Sentence['tokens']:
                tokens.append(token.replace(' ', ''))
            tfile.write(' '.join(tokens))
            tfile.write('\n')


def split_dataset(ipath, parts=10, n_train=9, n_test=1,
                  odir='expe/', shuffle=True):
    """Split the dataset into train/test + gold.

    Args:
        ipath: Path of the dataset.
        parts: Number of parts to split the dataset into.
        n_train: How many parts should be assigned to the training set.
        n_test: How many parts should be assigned to the testing set.
        odir: Output directory.
        shuffle: Whether the dataset should be shuffled.

    Returns:
        The sentence indexes of each dataset.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        tree = etree.parse(ifile)
    reviews = tree.getroot().getchildren()

    # Should we shuffle the dataset before breaking it into parts?
    if shuffle:
        random.shuffle(reviews)

    r = []
    s_idx = 0
    r_idx = 0
    for review in reviews:
        s_indexes = []
        for sentences in review.iterchildren():
            for sentence in sentences.iterchildren():
                s_indexes.append(s_idx)
                s_idx = s_idx + 1
        r.append([review, r_idx, s_indexes])
        r_idx = r_idx + 1
    reviews = r

    reviews_parts = list(split(reviews, parts))
    ret = []
    train_indexes = []
    test_indexes = []
    splitted_dataset = split_lst(reviews_parts, [n_train, n_test])
    for i in range(parts):
        (train_l, test_l) = splitted_dataset[i]
        train = []
        train_indexes.append([])
        for part in train_l:
            for (el, r_idx, s_idx) in part:
                train.append(el)
                train_indexes[i].extend(s_idx)

        gold = []
        test_indexes.append([])
        for part in test_l:
            for (el, r_idx, s_idx) in part:
                gold.append(el)
                test_indexes[i].extend(s_idx)

        test = deepcopy(gold)
        ret.append([train, test, gold])
        # Remove Opinions node in the test dataset
        for review in test:
            for sentences in review.iterchildren():
                for sentence in sentences.iterchildren():
                    for element in sentence.iterchildren():
                        if element.tag == 'Opinions':
                            sentence.remove(element)

        for (reviews, oname) in [(train, 'train.xml'),
                                 (test, 'test.xml'),
                                 (gold, 'gold.xml')]:
            root = etree.Element('Reviews')
            for review in reviews:
                root.append(review)

            if not os.path.exists(os.path.join(odir, str(i))):
                os.mkdir(os.path.join(odir, str(i)))
            opath = os.path.join(odir, str(i), oname)
            with open(opath, 'wb') as ofile:
                ofile.write(etree.tostring(root, encoding='UTF-8',
                                           pretty_print=True,
                                           xml_declaration=True))
    return train_indexes, test_indexes


def annotate_test_xml(ixml, oxml, pred_path, tokens_path, spans_path):
    """Annotate the XML from CRF labels.

    Args:
        ixml: Input XML file.
        oxml: Output XML file.
        pred_path: Txt file with labels.
        tokens_path: Txt file with tokens.
        spans_path: Txt file with token spans.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    # Open XML test file and read it.
    with codecs.open(ixml, 'r', 'utf-8') as ifile:
        root = etree.parse(ifile)
    # Get all Element_Sentence
    Sentences = []
    for review in root.getroot().iterchildren():
        for sentences in review.iterchildren():
            Sentences += sentences.getchildren()
    i_sentences = iter(Sentences)

    with codecs.open(pred_path, 'r', 'utf-8') as pfile, codecs.open(tokens_path, 'r', 'utf-8') as tfile, codecs.open(spans_path, 'r', 'utf-8') as sfile:
        sentence = i_sentences.next()
        opinions = etree.Element('Opinions')
        sentence.append(opinions)
        new_node = True
        for (pred, token, span) in zip(pfile, tfile, sfile):
            pred = pred.strip()
            token = token.strip()
            span = span.strip()

            if not pred:
                # Next sentence
                try:
                    new_node = True
                    sentence = i_sentences.next()
                    opinions = etree.Element('Opinions')
                    sentence.append(opinions)
                except StopIteration:
                    # No more sentence in this Review
                    break
            else:
                if pred == 'O':
                    new_node = True
                else:
                    # We've got a prediction!
                    if new_node:
                        new_node = False
                        # New prediction
                        opinion = etree.Element('Opinion')
                        opinion.set('polarity', '')
                        # opinion.set('category', pred)
                        opinion.set('category', '')
                        start = span.split(' ')[0]
                        end = span.split(' ')[1]
                        opinion.set('from', start)
                        opinion.set('to', end)
                        opinion.set('target', token)
                        opinions.append(opinion)
                    else:
                        # A continuation
                        end = span.split(' ')[1]
                        opinion.set('to', end)
                        old_target = opinion.get('target')
                        opinion.set('target', old_target + ' ' + token)

    with open(oxml, 'wb') as ofile:
        ofile.write(etree.tostring(root, encoding='UTF-8',
                                   pretty_print=True, xml_declaration=True))


def f_forms(ipath, opath):
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            for word in ifile:
                word = word.strip()
                if word:
                    # caps?
                    v = True in [c.isupper() for c in word]
                    ofile.write(str(v))
                    ofile.write('\t')
                    # all caps?
                    v = word.isupper()
                    ofile.write(str(v))
                    ofile.write('\t')
                    # Beg with caps?
                    v = word[0].isupper()
                    ofile.write(str(v))
                    ofile.write('\t')
                    # Has punc?
                    v = True in [c in string.punctuation for c in word]
                    ofile.write(str(v))
                    ofile.write('\t')
                    # all punc?
                    v = not (False in [c in string.punctuation for c in word])
                    ofile.write(str(v))
                    ofile.write('\t')
                    # has digit?
                    v = True in [c.isdigit() for c in word]
                    ofile.write(str(v))
                    ofile.write('\t')
                    # all digit?
                    v = word.isdigit()
                    ofile.write(str(v))
                ofile.write('\n')


def f_word_shape(ipath, opath):
    """Add word shape features from ipath to opath.

    Args:
        ipath: Input file with one word per line.
        opath: Output file with word shape instead of word.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            for word in ifile:
                word = word.strip()
                if word:
                    ofile.write(word_shape(word))
                ofile.write('\n')


def word_shape(word, compressed=True):
    """Return the compressed shape representation of word.

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1764467/ for a
 reference on compressed shape representation.

    All capitalized letters are replaced by 'A' and non-capitalized
    letters by 'a'.
    All digits are replaced by '0'.
    All other symbols are replaced by '_'.

    Args:
        word: Input word.
        compressed: Whether we should compress the representation or
        not.

    Returns:
        The compressed shape representation of word.
    """
    ret = ''
    if type(word) is not unicode:
        word = word.decode('utf-8')
    last = ''
    cur = ''
    for letter in word:
        last = cur
        if letter.isupper():
            cur = 'A'
        elif letter.islower():
            cur = 'a'
        elif letter.isdigit():
            cur = '0'
        else:
            cur = '_'
        if compressed and cur != last:
            ret += cur
    return ret


def f_stopwords(ipath, opath, stopwords=[]):
    """Extract stopwords information.

    Args:
        ipath: Inpu file with one word per line.
        opath: Output file TRUE if word is a stopword.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            for word in ifile:
                word = word.strip()
                if word:
                    if word.lower() in stopwords:
                        ofile.write('TRUE')
                    else:
                        ofile.write('FALSE')
                ofile.write('\n')


def f_senna(ipath, opath):
    """Extract POS/NER/CHK features with Senna.

    Args:
        ipath: Input file.
        opath: Feature output file.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    import data
    with open(ipath, 'rb') as ifile, open(opath, 'wb') as ofile:
        p = Popen(['./senna', '-notokentags', '-usrtokens', '-pos', '-chk', '-ner'],
                  stdin=ifile,
                  stdout=ofile, stderr=PIPE, cwd=data.SENNA_PATH)
        out, err = p.communicate()
        if p.returncode != 0:
            print(err.replace('*', '#'))


def f_senna_pos(ipath, opath):
    """Extract POS feature with Senna.

    Args:
        ipath: Input file.
        opath: Feature output file.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    import data
    with open(ipath, 'rb') as ifile, open(opath, 'wb') as ofile:
        p = Popen(['./senna', '-notokentags', '-usrtokens', '-pos'],
                  stdin=ifile,
                  stdout=ofile, stderr=PIPE, cwd=data.SENNA_PATH)
        out, err = p.communicate()
        if p.returncode != 0:
            print(err.replace('*', '#'))


def f_senna_ner(ipath, opath):
    """Extract NER feature with Senna.

    Args:
        ipath: Input file.
        opath: Feature output file.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    import data
    with open(ipath, 'rb') as ifile, open(opath, 'wb') as ofile:
        p = Popen(['./senna', '-notokentags', '-usrtokens', '-ner'],
                  stdin=ifile,
                  stdout=ofile, stderr=PIPE, cwd=data.SENNA_PATH)
        out, err = p.communicate()
        if p.returncode != 0:
            print(err.replace('*', '#'))
        

def f_senna_chk(ipath, opath):
    """Extract CHK feature with Senna.

    Args:
        ipath: Input file.
        opath: Feature output file.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    import data
    with open(ipath, 'rb') as ifile, open(opath, 'wb') as ofile:
        p = Popen(['./senna', '-notokentags', '-usrtokens', '-chk'],
                  stdin=ifile,
                  stdout=ofile, stderr=PIPE, cwd=data.SENNA_PATH)
        out, err = p.communicate()
        if p.returncode != 0:
            print(err.replace('*', '#'))


def f_bonsai(ipath, opath):
    """Extract POS features with BONSAI.

    Args:
        ipath: Input file.
        opath: Feature output file.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    import data
    temp_ofile = tempfile.NamedTemporaryFile(delete=False)
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        p = Popen(data.bonsai_cmd,
                  stdin=ifile,
                  stdout=temp_ofile, stderr=PIPE, env=os.environ, cwd=os.environ['BONSAI'])
        out, err = p.communicate()
        print(err)
        sys.stdout.flush()
        if p.returncode != 0:
            print(err.replace('*', '#'))
    temp_ofile.close()
    text_utils.cut(temp_ofile.name, [4], opath)
    # os.remove(temp_ofile.name)


def f_treetagger(ipath, opath, tagger):
    """Extract POS and LEMME features with TreeTagger.

    Args:
        ipath: Input file.
        opath: Feature output file.
        taglang: The lang for the tagger (en, fr).

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            words = []
            for word in ifile:
                word = word.strip()
                if word:
                    words.append(word)
                else:
                    tags = tagger.tag_text('\n'.join(words), tagonly=True)
                    tags2 = treetaggerwrapper.make_tags(tags)
                    for (word, pos, lemma) in tags2:
                        ofile.write(pos)
                        ofile.write('\t')
                        ofile.write(lemma)
                        ofile.write('\n')
                    ofile.write('\n')
                    words = []


def f_treetagger_pos(ipath, opath, tagger):
    """Extract POS feature with TreeTagger.

    Args:
        ipath: Input file.
        opath: Feature output file.
        taglang: The lang for the tagger (en, fr).

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            words = []
            for word in ifile:
                word = word.strip()
                if word:
                    words.append(word)
                else:
                    tags = tagger.tag_text('\n'.join(words), tagonly=True)
                    tags2 = treetaggerwrapper.make_tags(tags)
                    for (word, pos, lemma) in tags2:
                        ofile.write(pos)
                        ofile.write('\n')
                    ofile.write('\n')
                    words = []


def f_treetagger_lemme(ipath, opath, tagger):
    """Extract LEMME feature with TreeTagger.

    Args:
        ipath: Input file.
        opath: Feature output file.
        taglang: The lang for the tagger (en, fr).

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            words = []
            for word in ifile:
                word = word.strip()
                if word:
                    words.append(word)
                else:
                    tags = tagger.tag_text('\n'.join(words), tagonly=True)
                    tags2 = treetaggerwrapper.make_tags(tags)
                    for (word, pos, lemma) in tags2:
                        ofile.write(lemma)
                        ofile.write('\n')
                    ofile.write('\n')
                    words = []


def chunks(l, n):
    """ Yield n successive chunks from l.

    http://stackoverflow.com/a/2130042

    Args:
        l: The list to split.
        n: The number of chunks.

    Returns:
        Returns an iterator over the n chunks.
    """
    newn = int(1.0 * len(l) / n + 0.5)
    for i in xrange(0, n-1):
        yield l[i*newn:i*newn+newn]
    yield l[n*newn-newn:]


def split(a, n):
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


def split_lst(lst, splitter):
    """Split lst into the lists specified by splitter.

    E.g.
    >>> split_lst(range(5), [4, 1])
    [[[1, 2, 3, 4], [0]],
     [[2, 3, 4, 0], [1]],
     [[3, 4, 0, 1], [2]],
     [[4, 0, 1, 2], [3]],
     [[0, 1, 2, 3], [4]]]

    Args:
        lst: variable documentation.
        n: variable documentation.
        n_train: variable documentation.
        n_test: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    ret = []
    if len(lst) != sum(splitter):
        raise ValueError()
    for i in range(len(lst)):
        lst = rotate(lst, 1)
        seg = []
        acc = 0
        for s in splitter:
            seg.append(lst[acc:acc+s])
            acc += s
        ret.append(seg)

    return ret


def rotate(l, n):
    """Rotate a list by n elements."""
    return l[n:] + l[:n]


def merge_xml_files(files, opath):
    """Merge XML files assuming they have the same root element.

    Args:
        files: variable documentation.
        opath: variable documentation.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    root = None
    for ipath in files:
        with codecs.open(ipath, 'r', 'utf-8') as ifile:
            tree = etree.parse(ifile)
        if root is None:
            root = tree.getroot()
        else:
            for child in tree.getroot().iterchildren():
                root.append(child)
    with open(opath, 'wb') as ofile:
        ofile.write(etree.tostring(root, encoding='UTF-8',
                                   pretty_print=True,
                                   xml_declaration=True))


def f_lexicon(ipath, opath, lexicon, not_found=None, found=None):
    """Project lexicon on ipath.

    Args:
        ipath: variable documentation.
        lexicon: variable documentation.
        opath: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            for word in ifile:
                word = word.strip()
                if word:
                    word = word.lower()
                    if word in lexicon:
                        if found is None:
                            ofile.write(lexicon[word])
                        else:
                            ofile.write(found)
                    else:
                        if not_found is None:
                            ofile.write(word)
                        else:
                            ofile.write(not_found)
                ofile.write('\n')


def f_none(ipath, opath):
    None


def f_lexicon_multi_words(ipath, opath, lexicon, splitter=lambda s: s.split(),
                          not_found=None, found=None):
    """Project lexicon on ipath.

    Args:
        ipath: variable documentation.
        lexicon: variable documentation.
        opath: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile:
        with codecs.open(opath, 'w', 'utf-8') as ofile:
            sentence = []
            for word in ifile:
                word = word.strip()
                if word != '':
                    sentence.append(word.lower())
                else:
                    t_sentence = [None] * len(sentence)
                    j = 0
                    for entry in lexicon:
                        t_entry = splitter(entry)
                        t_entry_len = len(t_entry)
                        if t_entry_len == 0:
                            continue
                        t_words = []
                        for i in range(len(sentence)):
                            word = sentence[i]
                            t_words.append(word)
                            if t_words == t_entry:
                                if found is None:
                                    marker = lexicon[entry]
                                else:
                                    marker = found
                                for j in range(t_entry_len):
                                    t_sentence[i - j] = marker
                                t_words = []
                            if len(t_words) == t_entry_len:
                                t_words.pop(0)
                                idx = i - (t_entry_len - 1)
                                if t_sentence[idx] is None:
                                    if not_found is None:
                                        t_sentence[idx] = word
                                    else:
                                        t_sentence[idx] = 'NOT_FOUND'
                    for i in range(len(t_sentence)):
                        if t_sentence[i] is None:
                            if not_found is None:
                                t_sentence[i] = sentence[i]
                            else:
                                t_sentence[i] = 'NOT_FOUND'
                    ofile.write('\n'.join(t_sentence))
                    ofile.write('\n\n')
                    sentence = []


def read_bing_liu(neg_path, pos_path):
    """Return a dictionary of negative/positive words.

    Args:
        neg_path: variable documentation.
        pos_path: variable documentation.

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    for (path, c) in [(neg_path, 'negative'),
                      (pos_path, 'positive')]:
        with codecs.open(path, 'r', 'utf-8') as ifile:
            for word in ifile:
                word = word.strip()
                if word and not word.startswith(';'):
                    ret[word] = c
    return ret


def read_mpqa(mpqa_path):
    """Return a dictionary of negative/positive words.

    Args:
        neg_path: variable documentation.
        pos_path: variable documentation.

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(mpqa_path, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            cols = line.split()
            if len(cols) == 6:
                word = '='.join(cols[2].split('=')[1:])
                polarity = '='.join(cols[5].split('=')[1:])
                ret[word] = polarity
    return ret


def read_mpqa_plus(mpqa_path_plus):
    """Return a dictionary of negative/positive words.

    Args:

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(mpqa_path_plus, 'r', 'utf-8') as ifile:
        tree = etree.parse(ifile)
    root = tree.getroot()
    for lexical_entry in root.iterchildren():
        word = None
        polarity = None
        for el in lexical_entry.iterchildren():
            if el.tag == 'morpho':
                for node in el.iterchildren():
                    if node.tag == 'name':
                        word = node.text
            if el.tag == 'evaluation':
                polarity = el.get('subtype')
        if word is not None and polarity is not None:
            ret[word] = polarity
    return ret


def read_blogoscopie(path):
    """Return a dictionary of negative/positive words.

    Args:
        path: Path to file.

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(path, 'r', 'utf-8') as ifile:
        for line in ifile:
            line = line.strip()
            data = line.split('\t')
            ret[data[0]] = data[1]
    return ret


def read_lidilem(lidilem_path, pol_col):
    """Return a dictionary of negative/positive words.

    Format:
nom;domaine(s);sous-domaine(s);niveau de langue;intensité;polarité;fig/loc;

    Args:
        lidilem_path: Path to csv
        pol_col: The col containing the polarity.

     Returns:
        A dictionary of positive and negative words.

    Raises:
        IOError: An error occurred.
    """
    ret = {}
    with codecs.open(lidilem_path, 'r', 'utf-8') as ifile:
        first = True
        for line in ifile:
            if first:
                # skip first line with headers
                first = False
                continue
            line = line.strip()
            col = line.split(';')
            if col[pol_col] != '':
                ret[col[0]] = col[pol_col]
    return ret


def merge_pred_gold(pred_path, gold_path, opath):
    """Merge XML pred and gold XML.

    Args:
        pred_path: variable documentation.
        gold_path: variable documentation.
        opath: variable documentation.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(pred_path, 'r', 'utf-8') as ifile:
        p_root = etree.parse(ifile).getroot()
    with codecs.open(gold_path, 'r', 'utf-8') as ifile:
        g_root = etree.parse(ifile).getroot()

    for (p_review, g_review) in zip(p_root.iterchildren(),
                                    g_root.iterchildren()):
        for (p_sentences, g_sentences) in zip(p_review.iterchildren(),
                                              g_review.iterchildren()):
            for (p_sentence, g_sentence) in zip(p_sentences.iterchildren(),
                                                g_sentences.iterchildren()):
                for (p_element, g_element) in zip(p_sentence.iterchildren(),
                                                  g_sentence.iterchildren()):
                    if p_element.tag == 'Opinions':
                        for g_opinion in g_element.iterchildren():
                            if g_opinion.get('target') == 'NULL':
                                g_element.remove(g_opinion)
                        for p_opinion in p_element.iterchildren():
                            new_opinion = etree.Element('OpinionPred')
                            for attr in p_opinion.attrib:
                                new_opinion.set(attr, p_opinion.get(attr))
                            g_element.append(new_opinion)

    with open(opath, 'wb') as ofile:
        ofile.write(etree.tostring(g_root, encoding='UTF-8',
                                   pretty_print=True,
                                   xml_declaration=True))


def extract_errors_pred_gold(pred_path, gold_path, opath, sep=';'):
    """Extract errors in XML pred with gold XML.

    Args:
        pred_path: variable documentation.
        gold_path: variable documentation.
        opath: variable documentation.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(pred_path, 'r', 'utf-8') as ifile:
        p_root = etree.parse(ifile).getroot()
    with codecs.open(gold_path, 'r', 'utf-8') as ifile:
        g_root = etree.parse(ifile).getroot()

    with codecs.open(opath, 'w', 'utf-8') as ofile:
        for (p_review, g_review) in zip(p_root.iterchildren(),
                                        g_root.iterchildren()):
            for (p_sentences, g_sentences) in zip(p_review.iterchildren(),
                                                  g_review.iterchildren()):
                for (p_sentence, g_sentence) in zip(p_sentences.iterchildren(),
                                                    g_sentences.iterchildren()):
                    gold = []
                    pred = []
                    text = ''
                    for (p_element, g_element) in zip(p_sentence.iterchildren(),
                                                      g_sentence.iterchildren()):
                        if p_element.tag == 'text' and p_element.text is not None:
                            text = p_element.text
                        if p_element.tag == 'Opinions' and g_element.tag == 'Opinions':
                            for g_opinion in g_element.iterchildren():
                                if g_opinion.get('target') != 'NULL':
                                    gold.append(g_opinion.get('target'))
                            for p_opinion in p_element.iterchildren():
                                pred.append(p_opinion.get('target'))
                    if set(gold) != set(pred):
                        ofile.write('KO' + sep)
                    else:
                        ofile.write('OK' + sep)
                    ofile.write('%s%s%s%s%s\n' % (text.replace(';', '.,'),
                                                  sep, gold, sep, pred))


def generate_features(features, ident, workdir='./', stream=sys.stdout,
                      tokenizer=TweetTokenizer()):
    """Summary of the function

    Longer function information

    Args:
        features: variable documentation.
        workdir: variable documentation.

    Returns:
        The list of feature names generated.

    Raises:
        IOError: An error occurred.
    """
    f_names = []
    files = []
    stream.write('Tokenizing...' + '\n')
    stream.flush()
    sentences = extract_class_for_CRF(workdir + ident + '.xml',
                                      tokenizer=tokenizer)
    stream.write('Generating word and class files...' + '\n')
    stream.flush()
    write_class_for_CRF(sentences,
                        workdir + ident + '_w.txt',
                        workdir + ident + '_c.txt',
                        workdir + ident + '_s.txt')
    stream.write('Generating token file...' + '\n')
    stream.flush()
    write_tokenized_text(sentences, workdir + ident + '_tokens.txt')
    stream.write('Generating features files...' + '\n')
    stream.flush()
    for (feature_name, feature_function, features, input_name, parameters) in features:
        stream.write(feature_name + '...')
        stream.flush()
        f_names.extend(features)
        ipath = workdir + ident + '_' + input_name + '.txt'
        opath = workdir + ident + '_' + feature_name + '.txt'
        files.append(opath)
        feature_function(ipath=ipath, opath=opath, **parameters)
        stream.write('OK\n')
        stream.flush()
    stream.write('Generating model files...' + '\n')
    stream.flush()
    text_utils.paste(files,
                     workdir + ident + '_model.txt')
    stream.write('Removing feature files...' + '\n')
    stream.flush()
    for f in files:
        os.remove(f)
    return f_names


def fill_templates(builds, f_names, stream=sys.stdout, workdir='./'):
    """Fill templates file with indexes from f_names.

    Args:
        builds: variable documentation.
        f_names: variable documentation.
        workdir: variable documentation.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    stream.write('Filling templates...\n')
    stream.flush()
    for (ident, template_build) in builds:
        stream.write('    ' + ident + '\n')
        base_template = os.path.join(workdir, ident + '_template.txt')
        output = ''
        with codecs.open(base_template, 'r', 'utf-8') as ifile:
            for line in ifile:
                for (idx, f_name) in zip(range(len(f_names)), f_names):
                    line = line.replace(',' + f_name + ']', ',' + str(idx) + ']')
                    line = line.replace(',' + f_name + ',"', ',' + str(idx) + ',"')
                output = output + line
        with codecs.open(base_template, 'w', 'utf-8') as ofile:
            ofile.write(output)


def fill_template(ident, template, idir, f_names, stream=sys.stdout, workdir='./'):
    """Fill templates file with indexes from f_names.

    Args:
        ident:
        template:
        f_names: variable documentation.
        stream:
        workdir: variable documentation.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    stream.write('Filling template ' + template + '...\n' + '\n')
    with codecs.open(os.path.join(idir, template), 'r', 'utf-8') as ifile, codecs.open(os.path.join(workdir, template), 'w', 'utf-8') as ofile:
            for line in ifile:
                for (idx, f_name) in zip(range(len(f_names)), f_names):
                    line = line.replace(f_name, str(idx))
                ofile.write(line)


def pipeline_crf(builds, basedir, workdir, ret_file):
    """Summary of the function

    Longer function information

    Args:
        args: variable documentation.

    Returns:
        The f1 scores for each build.

    Raises:
        IOError: An error occurred.
    """
    scores = {}
    for (ident, template) in builds:
        train_file = workdir + 'train_model.txt'
        template_file = basedir + template
        test_file = workdir + 'test_model.txt'
        gold_xml = workdir + 'gold.xml'
        test_xml = workdir + 'test.xml'
        scores[ident] = pipeline(ident, template_file,
                                 train_file, test_file, gold_xml, test_xml,
                                 workdir=workdir)
    return scores


def train_model(ident, template, train_file,
                stream=sys.stdout, workdir='./'):
    """"Train a CRF model with Wapiti.

    Args:
        ident:
        template:
        train_file:
        stream:
        workdir:

    Returns:
        The path to the model trained.

    Raises:
        IOError: If the model file doesn't exist.
    """
    model_path = os.path.join(workdir, ident + '_model')
    stream.write('TRAINING ' + ident + '...\n' + '\n')
    import data
    cmd = data.wapiti_train_cmd + ['--pattern', template,
                                   train_file, model_path]
    stream.write(str(cmd) + '\n')
    p = Popen(cmd,
              stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    stream.write(out + '\n')
    stream.write(err + '\n')
    if not os.path.exists(model_path):
        stream.write(model_path + ' doesn\'t exist\n' + '\n')
        stream.write(ident + '\n')
        stream.write(template + '\n')
        stream.write(train_file + '\n')
        raise IOError()
    return model_path


def annotate(ident, model_path, test_file, test_xml,
             word_file, spans_file,
             stream=sys.stdout, workdir='./'):
    """Annotate a test file and produce the final XML.

    Args:
        model_path:
        test_file:
        stream:
        workdir:

    Returns:
        The path to the XML produced.

    Raises:
        IOError: An error occurs.
"""
    pred_file = os.path.join(workdir, ident + '_pred.txt')
    pred_xml = os.path.join(workdir, ident + '_pred.xml')
    stream.write('ANNOTATING...\n' + '\n')
    import data
    cmd = data.wapiti_label_cmd + ['--model', model_path, test_file, pred_file]

    p = Popen(cmd,
              stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    stream.write(out + '\n')
    stream.write(err + '\n')
    if not os.path.exists(pred_file):
        stream.write(pred_file + ' doesn\'t exist\n' + '\n')
        raise IOError()
    annotate_test_xml(test_xml,
                      pred_xml,
                      pred_file,
                      word_file, spans_file)
    if not os.path.exists(pred_xml):
        stream.write(pred_xml + ' doesn\'t exist\n' + '\n')
        raise IOError()
    return pred_xml


def evaluate(ident, pred_xml, gold_xml, stream=sys.stdout):
    """"""
    import data
    stream.write('EVALUATING...\n' + '\n')
    p = Popen(['java', '-cp', './A.jar', 'absa16.Do', 'Eval',
               '-prd', os.path.abspath(pred_xml),
               '-gld', os.path.abspath(gold_xml),
               '-evs', '2', '-phs', 'A', '-sbt', 'SB1'],
              stdout=PIPE, stderr=PIPE,
              cwd=os.path.join(data.DATA_DIR, 'SEMEVAL16/ABSA/absa16_evaluation'))
    out, err = p.communicate()
    stream.write(out + '\n')
    stream.write(err + '\n')
    scores = {}
    for line in out.split():
        line = line.strip()
        if re.search('^(F-MEASURE|PRE|REC)', line):
            obj, score = line.split('=')
            score = float(score)
            if math.isnan(float(score)):
                score = 0
            scores[obj] = float(score)
    return scores


def pipeline(ident, template_file,
             train_file, test_file, gold_xml, test_xml,
             stream=sys.stdout, workdir='./'):
    """Small pipeline.

    Args:
        ident: variable documentation.
        template: variable documentation.
        train_file: variable documentation.
        test_file: variable documentation.
        gold_xml: variable documentation.
        stream: variable documentation.
        workdir: variable documentation.

    Returns:
        The scores for each build.

    Raises:
        IOError: An error occurred.
    """
    word_file = os.path.join(workdir, 'test_w.txt')
    spans_file = os.path.join(workdir, 'test_s.txt')
    model_path = train_model(ident, template_file, train_file, stream, workdir)
    pred_xml = annotate(ident, model_path, test_file, test_xml,
                        word_file, spans_file,
                        stream, workdir)
    return evaluate(ident, pred_xml, gold_xml, stream)


def split_data(train_xml, feature_files, parts, n_train, n_test):
    """Summary of the function

    Args:
        train_xml: variable documentation.
        feature_files: variable documentation.
        parts: variable documentation.
        n_train: variable documentation.
        n_test: variable documentation.

    Returns:
        Returns information

    Raises:
        IOError: An error occurred.
    """
    pass


def extract_sentences(ipath, opath, sentences_idx):
    """Extract sentences in ipath by the given idx.

    Args:
        ipath: variable documentation.
        opath: variable documentation.
        sentences_idx: variable documentation.

    Returns:
        Nothing.

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(ipath, 'r', 'utf-8') as ifile, codecs.open(opath, 'w', 'utf-8') as ofile:
        idx = 0
        cur_sent = []
        for line in ifile:
            line = line.strip()
            if line == '':
                idx = idx + 1
                if len(cur_sent) > 0:
                    ofile.write('\n'.join(cur_sent))
                    ofile.write('\n\n')
                cur_sent = []
            elif idx in sentences_idx:
                cur_sent.append(line)


class TreeTaggerTokenizer(nltk.tokenize.api.StringTokenizer):
    """Tokenize a text using a TreeTaggerWrapper tagger.

    Attributes:
        tagger: The tagger.
    """

    def __init__(self, tagger):
        """Init TreeTaggerTokenizer with the tagger.
        """
        self.tagger = tagger

    def tokenize(self, text):
        if isinstance(text, str):
            text = text.decode('utf-8')
        ret = self.tagger.tag_text(text, prepronly=True,
                                   tagblanks=False, notagurl=True, notagemail=True,
                                   notagip=True, notagdns=True, nosgmlsplit=True)
        return ret


def generate_template(template_build, f_names, output):
    """Generate a template file from the template_build specifications.

    The format of the template specification is the following :
    template_build := [spec...]
    spec           := (fmt, pos)
    fmt            := a format string used to generate features
    pos            := an array of positions (one line generated per position)

    For example :
    >>> tb = [('U:word=%X[POS,WORD]', [-2, -1, 0, 1, 2]),
              ('U:pre=%m[POS,WORD,"^.?"]', [0]),
              'U:Caps? L=%t[-1,WORD,"\u"]', 'B']
    >>> crf_absa16.generate_template(tb, ['WORD'], '/tmp/test.txt')
# BEGIN test.txt
U:0_word=%X[0,0]
U:1_0_word=%X[0,0]
U:2_1_0_word=%X[0,0]
U:3_2_1_0_word=%X[0,0]
U:4_3_2_1_0_word=%X[0,0]

U:5_pre=%m[0,0,"^.?"]

U:Caps? L=%t[-1,0,"\u"]

B
B
# END test.txt

    Args:
        template_build: The specification of the template.
        f_names: Index of features name.
        output: Output file.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    with codecs.open(output, 'w', 'utf-8') as ofile:
        spec_idx = 0
        for spec in template_build:
            if isinstance(spec, str):
                fmt = spec
                for (f_pos, f_name) in zip(range(len(f_names)), f_names):
                    fmt = fmt.replace(',' + f_name + ']', ',' + str(f_pos) + ']')
                    fmt = fmt.replace(',' + f_name + ',"', ',' + str(f_pos) + ',"')
                spec_idx += 1
                ofile.write(fmt)
            else:
                fmt_type, fmt_name, fmt_cmd, fmt_f_name, positions = spec
                for pos in positions:
                    fmt = '%s:%d_%s=%%%s[%d,%s]\n' % (fmt_type, spec_idx, fmt_name, fmt_cmd, pos, fmt_f_name)
                    for (f_pos, f_name) in zip(range(len(f_names)), f_names):
                        fmt = fmt.replace(',' + f_name + ']', ',' + str(f_pos) + ']')
                        fmt = fmt.replace(',' + f_name + ',"', ',' + str(f_pos) + ',"')
                    spec_idx += 1
                    ofile.write(fmt)
            ofile.write('\n')


def generate_templates(builds, f_names, stream=sys.stdout, workdir='./'):
    """Generate all templates in build.

    Args:
        builds: List of tuples (ident, template_build)
        f_names: Index of features name.
        workdir: Output directory.

    Returns:
        Nothing

    Raises:
        IOError: An error occurred.
    """
    for (ident, template_build) in builds:
        stream.write('Generating template %s' % ident)
        output_file = os.path.join(workdir, ident + '_template.txt')
        generate_template(template_build, f_names, output_file)
