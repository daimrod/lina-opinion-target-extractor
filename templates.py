en_lexicons = [('U', 'bing_liu', 'X', 'BING_LIU', [-2, -1, 0, 1, 2]),
               ('U', 'mpqa', 'X', 'MPQA', [-2, -1, 0, 1, 2]),
               ('U', 'mpqa_plus', 'X', 'MPQA_PLUS', [-2, -1, 0, 1, 2])]

fr_lexicons = [('U', 'lidi_verbes', 'X', 'LIDI_VERBES', [-2, -1, 0, 1, 2]),
               ('U', 'lidi_noms', 'X', 'LIDI_NOMS', [-2, -1, 0, 1, 2]),
               ('U', 'lidi_adjectifs', 'X', 'LIDI_ADJECTIFS', [-2, -1, 0, 1, 2]),
               ('U', 'blogoscopie', 'X', 'BLOGOSCOPIE', [-2, -1, 0, 1, 2])]

form1 = [('U', 'word', 'X', 'WORD', [-2, -1, 0, 1, 2]),
         ('U', 'word_shape', 'X', 'WORD_SHAPE', [-2, -1, 0, 1, 2]),
         'U:Pre_1 X=%m[ 0,WORD,"^.?"]',
         'U:Pre_2 X=%m[ 0,WORD,"^.?.?"]',
         'U:Pre_3 X=%m[ 0,WORD,"^.?.?.?"]',
         'U:Pre_4 X=%m[ 0,WORD,"^.?.?.?.?"]',
         'U:Suf_1 X=%m[ 0,WORD,".?$"]',
         'U:Suf_2 X=%m[ 0,WORD,".?.?$"]',
         'U:Suf_3 X=%m[ 0,WORD,".?.?.?$"]',
         'U:Suf_4 X=%m[ 0,WORD,".?.?.?.?$"]']

form2 = [('U', 'word', 'X', 'WORD', [-2, -1, 0, 1, 2]),
         ('U', 'word_shape', 'X', 'WORD_SHAPE', [-2, -1, 0, 1, 2]),
         'U:Pre_1 X=%m[ 0,WORD,"^.?"]',
         'U:Pre_2 X=%m[ 0,WORD,"^.?.?"]',
         'U:Pre_3 X=%m[ 0,WORD,"^.?.?.?"]',
         'U:Pre_4 X=%m[ 0,WORD,"^.?.?.?.?"]',
         'U:Suf_1 X=%m[ 0,WORD,".?$"]',
         'U:Suf_2 X=%m[ 0,WORD,".?.?$"]',
         'U:Suf_3 X=%m[ 0,WORD,".?.?.?$"]',
         'U:Suf_4 X=%m[ 0,WORD,".?.?.?.?$"]',
         'U:Caps? L=%t[-1,WORD,"\u"]',
         'U:Caps? X=%t[ 0,WORD,"\u"]',
         'U:Caps? R=%t[ 1,WORD,"\u"]',
         'U:AllC? X=%t[ 0,WORD,"^\u*$"]',
         'U:BegC? X=%t[ 0,WORD,"^\u"]',
         'U:Punc? L=%t[-1,WORD,"\p"]',
         'U:Punc? X=%t[ 0,WORD,"\p"]',
         'U:Punc? R=%t[ 1,WORD,"\p"]',
         'U:AllP? X=%t[ 0,WORD,"^\p*$"]',
         'U:InsP? X=%t[ 0,WORD,".\p."]',
         'U:Numb? L=%t[-1,WORD,"\d"]',
         'U:Numb? X=%t[ 0,WORD,"\d"]',
         'U:Numb? R=%t[ 1,WORD,"\d"]',
         'U:AllN? X=%t[ 0,WORD,"^\d*$"]']

stop_words = [('U', 'stop_words', 'X', 'STOP_WORDS', [-2, -1, 0, 1, 2])]

pos = [('U', 'pos', 'X', 'POS', [-2, -1, 0, 1, 2])]
morph = pos + [('U', 'chk', 'X', 'CHK', [-2, -1, 0, 1, 2]),
               ('U', 'ner', 'X', 'NER', [-2, -1, 0, 1, 2])]


full = form2 + stop_words + morph + ['B']

en_all = form2 + stop_words + morph + en_lexicons + ['B']
en_all_old = ['''U:word_LL=%X[-2,WORD]
U:word__L=%X[-1,WORD]
U:word__X=%X[ 0,WORD]
U:word__R=%X[ 1,WORD]
U:word_RR=%X[ 2,WORD]

U:pos_LL=%X[-2,POS]
U:pos__L=%X[-1,POS]
U:pos__X=%X[ 0,POS]
U:pos__R=%X[ 1,POS]
U:pos_RR=%X[ 2,POS]

U:chk_LL=%X[-2,CHK]
U:chk__L=%X[-1,CHK]
U:chk__X=%X[ 0,CHK]
U:chk__R=%X[ 1,CHK]
U:chk_RR=%X[ 2,CHK]

U:ner_LL=%X[-2,NER]
U:ner__L=%X[-1,NER]
U:ner__X=%X[ 0,NER]
U:ner__R=%X[ 1,NER]
U:ner_RR=%X[ 2,NER]

U:word_shape_LL=%X[-2,WORD_SHAPE]
U:word_shape__L=%X[-1,WORD_SHAPE]
U:word_shape__X=%X[ 0,WORD_SHAPE]
U:word_shape__R=%X[ 1,WORD_SHAPE]
U:word_shape_RR=%X[ 2,WORD_SHAPE]

U:stop_words_LL=%X[-2,STOP_WORDS]
U:stop_words__L=%X[-1,STOP_WORDS]
U:stop_words__X=%X[ 0,STOP_WORDS]
U:stop_words__R=%X[ 1,STOP_WORDS]
U:stop_words_RR=%X[ 2,STOP_WORDS]

U:bing_liu_LL=%X[-2,BING_LIU]
U:bing_liu__L=%X[-1,BING_LIU]
U:bing_liu__X=%X[ 0,BING_LIU]
U:bing_liu__R=%X[ 1,BING_LIU]
U:bing_liu_RR=%X[ 2,BING_LIU]

U:mpqa_LL=%X[-2,MPQA]
U:mpqa__L=%X[-1,MPQA]
U:mpqa__X=%X[ 0,MPQA]
U:mpqa__R=%X[ 1,MPQA]
U:mpqa_RR=%X[ 2,MPQA]

U:mpqa_plus_LL=%X[-2,MPQA_PLUS]
U:mpqa_plus__L=%X[-1,MPQA_PLUS]
U:mpqa_plus__X=%X[ 0,MPQA_PLUS]
U:mpqa_plus__R=%X[ 1,MPQA_PLUS]
U:mpqa_plus_RR=%X[ 2,MPQA_PLUS]

U:Pre-1 X=%m[ 0,WORD,"^.?"]
U:Pre-2 X=%m[ 0,WORD,"^.?.?"]
U:Pre-3 X=%m[ 0,WORD,"^.?.?.?"]
U:Pre-4 X=%m[ 0,WORD,"^.?.?.?.?"]

U:Suf-1 X=%m[ 0,WORD,".?$"]
U:Suf-2 X=%m[ 0,WORD,".?.?$"]
U:Suf-3 X=%m[ 0,WORD,".?.?.?$"]
U:Suf-4 X=%m[ 0,WORD,".?.?.?.?$"]

U:Caps? L=%t[-1,0,"\u"]
U:Caps? X=%t[ 0,0,"\u"]
U:Caps? R=%t[ 1,0,"\u"]
U:AllC? X=%t[ 0,0,"^\u*$"]
U:BegC? X=%t[ 0,0,"^\u"]
U:Punc? L=%t[-1,0,"\p"]
U:Punc? X=%t[ 0,0,"\p"]
U:Punc? R=%t[ 1,0,"\p"]
U:AllP? X=%t[ 0,0,"^\p*$"]
U:InsP? X=%t[ 0,0,".\p."]
U:Numb? L=%t[-1,0,"\d"]
U:Numb? X=%t[ 0,0,"\d"]
U:Numb? R=%t[ 1,0,"\d"]
U:AllN? X=%t[ 0,0,"^\d*$"]

B
''']
en_all_form = stop_words + morph + en_lexicons + ['B']
en_all_lex = form2 + stop_words + morph + ['B']
en_all_morph = form2 + stop_words + en_lexicons + ['B']
en_morph = pos + ['B']
en_form = form1 + ['B']
en_morph_lex = pos + en_lexicons + ['B']

en_all = form2 + stop_words + morph + en_lexicons + ['B']
en_all_form = stop_words + morph + en_lexicons + ['B']
en_all_lex = form2 + stop_words + morph + ['B']
en_morph = pos + ['B']
en_form = form1 + ['B']
en_morph_lex = pos + en_lexicons + ['B']

fr_all = form2 + stop_words + morph + fr_lexicons + ['B']
fr_all_form = stop_words + morph + fr_lexicons + ['B']
fr_all_lex = form2 + stop_words + morph + ['B']
fr_all_morph = form2 + stop_words + fr_lexicons + ['B']
fr_morph = pos + ['B']
fr_form = form1 + ['B']
fr_morph_lex = pos + fr_lexicons + ['B']
