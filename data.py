#!/usr/bin/env python
# -*- coding: utf-8 -*-

import crf_absa16
import pickle
import json
import os

if 'DATA_DIR' in os.environ:
    DATA_DIR = os.environ['DATA_DIR']
else:
    DATA_DIR = '/media/jadi-g/DATADRIVE1/corpus'

# Executables

## Wapiti Configuration
WAPITI_PATH = os.path.expanduser('~/src/c/Wapiti/wapiti')
wapiti_train_cmd = [WAPITI_PATH,
                    'train',
                    '--nthread', '4',
                    # , '--rho1', '0' # 0 = no l1 reg
                    # , '--rho2', '1' # 0 = no l2 reg
                    # , '--maxiter', '10' # 0 = unlimited
                    # , '--histsz', '6'
                    # , '--stopeps', '0.00001' # stoping criteriong
                    # , '--objwin', '10'
                    # , '--maxls', '20'
                    ]

wapiti_label_cmd = [WAPITI_PATH, 'label', '--label']

## BONSAI Configuration
BONSAI_PATH = os.path.expanduser('~/src/thesis/bonsai_v3.2/')
os.environ['BONSAI'] = BONSAI_PATH
bonsai_cmd = [os.path.join(BONSAI_PATH, 'bin/bonsai_bky_parse.sh'),
              '-n', '-f', 'udep']

## TreeTagger Configuration
TREETAGGER_PATH = os.path.expanduser('~/src/thesis/treetagger/')
os.environ['TAGDIR'] = TREETAGGER_PATH
os.environ['TAGOPT'] = '-token -lemma -sgml -quiet'
os.environ['TAGINENC'] = 'utf8'
os.environ['TAGOUTENC'] = 'utf8'
os.environ['TAGINENCERR'] = 'ignore'

## Senna Configuration
SENNA_PATH = os.path.expanduser('~/src/thesis/senna/')

# Files

restaurants_en_2016 = os.path.join(DATA_DIR, 'SEMEVAL16/ABSA/TRAIN/RESTAURANTS/SUB1/ABSA16_Restaurants_Train_SB1_v2.xml')
restaurants_en_2015 = os.path.join(DATA_DIR, 'SEMEVAL15/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml')
restaurants_en_test_2015 = os.path.join(DATA_DIR, 'SEMEVAL15/ABSA15_Restaurants_Test.xml')
restaurants_en_gold_2015 = os.path.join(DATA_DIR, 'SEMEVAL15/ABSA15_Restaurants_Gold.xml')

restaurants_fr_2016 = os.path.join(DATA_DIR, 'SEMEVAL16/ABSA/TRAIN/RESTAURANTS/SUB1/ABSA16FR-RestaurantsTrain/ABSA16FR_Restaurants_Train-withcontent.xml')

# Lexicons

bing_liu_lexicon_path = dict({'negative': os.path.join(DATA_DIR, 'bing-liu-lexicon/negative-words.txt'),
                              'positive': os.path.join(DATA_DIR, 'bing-liu-lexicon/positive-words.txt')})
bing_liu_lexicon = crf_absa16.read_bing_liu(bing_liu_lexicon_path['negative'], bing_liu_lexicon_path['positive'])

mpqa_lexicon_path = os.path.join(DATA_DIR, 'subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')
mpqa_lexicon = crf_absa16.read_mpqa(mpqa_lexicon_path)

mpqa_plus_lexicon_path = os.path.join(DATA_DIR, 'mpqa_plus_lex.xml')
mpqa_plus_lexicon = crf_absa16.read_mpqa_plus(mpqa_plus_lexicon_path)

lidilem_base_path = os.path.join(DATA_DIR, 'LIDILEM-Grenoble-lexiques')

lidilem_adjectifs_path = os.path.join(lidilem_base_path, 'Adjectifs.csv')
lidilem_adjectifs = crf_absa16.read_lidilem(lidilem_adjectifs_path, 7)

lidilem_noms_path = os.path.join(lidilem_base_path, 'Noms.csv')
lidilem_noms = crf_absa16.read_lidilem(lidilem_noms_path, 5)

lidilem_verbes_path = os.path.join(lidilem_base_path, 'Verbes.csv')
lidilem_verbes = crf_absa16.read_lidilem(lidilem_verbes_path, 8)

blogoscopie_path = os.path.join(DATA_DIR, 'blogoscopie/francais/LEXIQUE_EVALUATION_INITIAL_STRIPPED.txt')
blogoscopie = crf_absa16.read_blogoscopie(blogoscopie_path)

# VSM

DIM2WORD_PATH = os.path.join(DATA_DIR, 'expe_fastcluster', 'toto2.dim2word.json')
with open(DIM2WORD_PATH, 'rb') as ifile:
    dim2word = json.load(ifile)
    dict_dim2word = {}
for i in range(len(dim2word)):
    dict_dim2word[dim2word[i]] = i

LINKAGE_ID = 7900173
MODELES = ['average', 'centroid', 'complete', 'median', 'single', 'ward', 'weighted']
MODELE_PATH_FMT = os.path.join(DATA_DIR, 'expe_fastcluster/30961x30961-%s-None-diag-lm.json')

