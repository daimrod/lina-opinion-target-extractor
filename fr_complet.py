#!/usr/bin/env python
# -*- coding: utf-8 -*-

import crf_absa16
import sys
import run
import data
import treetaggerwrapper
from nltk.corpus import stopwords
import templates

PARTS, WORKDIR = run.init(sys.argv)

TRAIN_PATH = data.restaurants_fr_2016
TEST_PATH = None
GOLD_PATH = None

BUILDS = []

BUILDS = [('fr_all', templates.fr_all),
          ('fr_all-form', templates.fr_all_form),
          ('fr_all-lex', templates.fr_all_lex),
          ('fr_all-morph', templates.fr_all_morph),
          ('fr_morph', templates.fr_morph),
          ('fr_form', templates.fr_form),
          ('fr_morph_lex', templates.fr_morph_lex)]

NOT_FOUND = None
FOUND = None


tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
tokenizer = crf_absa16.TreeTaggerTokenizer(tagger)
FEATURES = [
    ['word_shape', crf_absa16.f_word_shape, ['WORD_SHAPE'], 'w', dict()],
    ['stop_words', crf_absa16.f_stopwords, ['STOP_WORDS'], 'w', dict(stopwords=stopwords.words('french'))],
    ['senna', crf_absa16.f_senna, ['SENNA_POS', 'CHK', 'NER'], 'tokens', dict()],
    ['pos', crf_absa16.f_treetagger_pos, ['POS'], 'w', dict(tagger=tagger)],
    ['lemme', crf_absa16.f_treetagger_lemme, ['LEMME'], 'w', dict(tagger=tagger)],
    ['lidi_adjectifs', crf_absa16.f_lexicon, ['LIDI_ADJECTIFS'], 'w',
     dict(lexicon=data.lidilem_adjectifs, not_found=NOT_FOUND, found=FOUND)],
    ['lidi_noms', crf_absa16.f_lexicon, ['LIDI_NOMS'], 'w',
     dict(lexicon=data.lidilem_noms, not_found=NOT_FOUND, found=FOUND)],
    ['lidi_verbes', crf_absa16.f_lexicon, ['LIDI_VERBES'], 'w',
     dict(lexicon=data.lidilem_verbes, not_found=NOT_FOUND, found=FOUND)],
    ['blogoscopie', crf_absa16.f_lexicon_multi_words, ['BLOGOSCOPIE'], 'w',
     dict(lexicon=data.blogoscopie, splitter=tokenizer.tokenize, not_found=NOT_FOUND, found=FOUND)],
]

FEATURES = [['w', crf_absa16.f_none, ['WORD'], 'w', dict()]] + FEATURES
FEATURES = FEATURES + [['c', crf_absa16.f_none, ['CLASS'], 'c', dict()]]

SHUFFLE = False

run.full_pipeline_with_fold(PARTS, WORKDIR,
                            BUILDS, FEATURES, SHUFFLE,
                            TRAIN_PATH, TEST_PATH, GOLD_PATH,
                            tokenizer=tokenizer)
