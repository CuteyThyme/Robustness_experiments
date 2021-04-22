# coding=utf-8
import os
from fastNLP.embeddings.char_embedding import CNNCharEmbedding, LSTMCharEmbedding
from fastNLP.embeddings import Embedding, StaticEmbedding, BertEmbedding, StackEmbedding, ElmoEmbedding, RobertaEmbedding
from fastNLP.io.pipe.conll import Conll2003NERPipe, OntoNotesNERPipe
from fastNLP.core import DataSet, Instance
from config import Config

config = Config()
normalize_embed = True
type = ['train', 'test', 'dev', 'oov', 'concat', 'crossCategory', 'toLonger', 'entityTyposSwap', 'addAdverb', 'addLrrSent', 'caseLower', 
                  'caseTitle', 'caseUpper', 'contraction', 'keyboard', 'mlm', 'number', 'ocr', 'punctuationAddBracket', 'reverseNeg',
                  'spelling', 'tense', 'twitterRandom', 'typosRandom', 'wordEmbedding', 'wordNetAntonym', 'wordNetSynonym',
       'ori_oov', 'ori_concat', 'ori_crossCategory', 'ori_toLonger', 'ori_entityTyposSwap', 'ori_addAdverb', 'ori_addLrrSent', 'ori_caseLower', 
        'ori_caseTitle', 'ori_caseUpper', 'ori_contraction', 'ori_keyboard', 'ori_mlm', 'ori_number', 'ori_ocr', 'ori_punctuationAddBracket', 
        'ori_reverseNeg', 'ori_spelling', 'ori_tense', 'ori_twitterRandom', 'ori_typosRandom', 'ori_wordEmbedding', 'ori_wordNetAntonym', 'ori_wordNetSynonym']

def load_data(dataset, word_embedding, char_embedding):
    if dataset == 'conll2003':
        conll_dir = './data/conll2003/'
        file_name = [conll_dir + t + '.txt' for t in type]
    
    elif dataset == 'en-ontonotes':
        ontonotes_dir = './data/en-ontonotes/'
        file_name = [ontonotes_dir + t + '.txt' for t in type]
        config.update(lr=0.0001)
        config.update(optim_type='sgd')
        config.update(trans_dropout=0.15)
        config.update(batch_size=16)

    elif dataset == 'ace':
        ace_dir = './data/ace/'
        file_name = [ace_dir + t + '.txt' for t in type]

    paths = dict(zip(type, file_name))
    data = Conll2003NERPipe(encoding_type=config.encoding_type).process_from_file(paths)


    if char_embedding == 'cnn':
        char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=100, char_emb_size=100, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='avg'
                                      , include_word_start_end=True, min_char_freq=2)
    
    if word_embedding == 'rand':
        word_embed = StaticEmbedding(vocab=data.get_vocab('words'), model_dir_or_name=None, embedding_dim=100)

    elif word_embedding == 'glove':    
        word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                 model_dir_or_name='en-glove-6b-100d', embedding_dim=100,
                                 requires_grad=False, lower=False, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=False)
        
    elif word_embedding == 'elmo':
        word_embed = ElmoEmbedding(vocab=data.get_vocab('words'), model_dir_or_name='en-original', layers='mix', 
                                    requires_grad=False, word_dropout=0.0, dropout=0.5, cache_word_reprs=False)
        word_embed.set_mix_weights_requires_grad()

    elif word_embedding == 'bert':
        word_embed = BertEmbedding(vocab=data.get_vocab('words'), model_dir_or_name='en-large-uncased', layers='4,-2,-1',
                                  requires_grad=False, word_dropout=0, dropout=0.2)
        

    if char_embedding is not None and word_embedding is not None:
        embed = StackEmbedding([char_embed, word_embed], dropout=0, word_dropout=0.02)

    elif char_embedding is not None:
        embed = char_embed

    elif word_embedding is not None:
        embed = word_embed

    return data, embed

