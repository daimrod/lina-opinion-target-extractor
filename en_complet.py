#!/usr/bin/env python
# -*- coding: utf-8 -*-

import crf_absa16
import sys
import run
import data
import treetaggerwrapper
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import templates

PARTS, WORKDIR = run.init(sys.argv)

# Pipeline description begins here !

TRAIN_PATH = data.restaurants_en_2015
TEST_PATH = data.restaurants_en_test_2015
GOLD_PATH = data.restaurants_en_gold_2015

BUILDS = [('en_all', templates.en_all),
          ('en_all_old', templates.en_all_old),
          ('en_all-form', templates.en_all_form),
          ('en_all-lex', templates.en_all_lex),
          ('en_all-morph', templates.en_all_morph),
          ('en_morph', templates.en_morph),
          ('en_form', templates.en_form),
          ('en_morph_lex', templates.en_morph_lex)]

NOT_FOUND = None
FOUND = None

tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
# tokenizer = crf_absa16.TreeTaggerTokenizer(tagger)
tokenizer = TweetTokenizer()

FEATURES = [
    ['word_shape', crf_absa16.f_word_shape, ['WORD_SHAPE'], 'w', dict()],
    ['stop_words', crf_absa16.f_stopwords, ['STOP_WORDS'], 'w', dict(stopwords=[])],
    # ['stop_words', crf_absa16.f_stopwords, ['STOP_WORDS'], 'w', dict(stopwords=stopwords.words('english'))],
    ['senna', crf_absa16.f_senna, ['POS', 'CHK', 'NER'], 'tokens', dict()],
    ['lemme', crf_absa16.f_treetagger_lemme, ['LEMME'], 'w', dict(tagger=tagger)],
    ['bing_liu', crf_absa16.f_lexicon, ['BING_LIU'], 'w', dict(lexicon=data.bing_liu_lexicon,
                                                               not_found=NOT_FOUND,
                                                               found=FOUND)],
    ['mpqa', crf_absa16.f_lexicon, ['MPQA'], 'w', dict(lexicon=data.mpqa_lexicon,
                                                       not_found=NOT_FOUND,
                                                       found=FOUND)],
    ['mpqa_plus', crf_absa16.f_lexicon_multi_words, ['MPQA_PLUS'], 'w', dict(lexicon=data.mpqa_plus_lexicon,
                                                                             not_found=NOT_FOUND,
                                                                             found=FOUND)],
]

FEATURES = [['w', crf_absa16.f_none, ['WORD'], 'w', dict()]] + FEATURES
FEATURES = FEATURES + [['c', crf_absa16.f_none, ['CLASS'], 'c', dict()]]

SHUFFLE = False

# Pipeline description ends here !

run.full_pipeline_with_fold(PARTS, WORKDIR,
                            BUILDS, FEATURES, SHUFFLE,
                            TRAIN_PATH, TEST_PATH, GOLD_PATH,
                            tokenizer=tokenizer)
