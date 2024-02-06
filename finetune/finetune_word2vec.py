import argparse
import pandas as pd
import numpy as np

from functools import reduce
from copy import copy

import os
import timeit
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

import pyarrow.parquet as pq
from konlpy.tag import Mecab
# from torch import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

from utils import *

mecab = Mecab()
seed_everything(42)

def get_config():
    p = argparse.ArgumentParser(description="Set arguments.")

    p.add_argument("--seed", default="42", type=int)
    p.add_argument("--rank", default="10", type=int)
    p.add_argument("--epoch", default="30", type=int)
    p.add_argument("--score", default="0.2", type=float)
    p.add_argument("--model_fn", default="word2vec_gensim.model", type=str)
    p.add_argument("--terms", default="ego_terms", type=str)             # ego_terms, refined_ego_terms
    p.add_argument("--vocab_word", default="인공지능", type=str)
    p.add_argument("--input_text", default="인공지능과 자동차", type=str)
    p.add_argument("--dir_path", default="/data/nevret/word-embedding-model", type=str)
    
    config = p.parse_args()

    return config

def remove_stopwords(config):
    stopword_list = get_stopwords(config)
    col = [i for i in col if i not in stopword_list]

    # remove_set = {'및','할','수','본','그','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','\n'}
    # col = [i for i in col if i not in remove_set]

    return col

def read_data(config):
    data = pq.read_table(os.path.join(config.dir_path + f'/finetune/data/tipa_text_tokens_20220328.parquet')).to_pandas().sort_values('sbjt_id').reset_index(drop=True)
    data = data.rename(columns={'terms': 'ego_terms',
                                'refined_terms': 'refined_ego_terms'})
    
    # data['ego_terms'] = data['ego_terms'].apply(remove_stopwords)
    
    return data

def get_stopwords(config):
    stopwords_file = list(open(os.path.join(config.dir_path, 'finetune/stopwords.txt'), 'r'))
    list_stopword = []
    for i in stopwords_file:
        list_stopword.append(i[:-1])

    return list_stopword

def tokenized_mecab(data):
    result = mecab.morphs(data)
    result = ' '.join(result)
    
    return result

def input2vec(config, model, stopword_list):    
    def out_stopwords(data, list_stopwords):
        data = data.split(' ')
        data = [token for token in data if token not in list_stopwords]
        return data

    word_dict = {}
    for vocab in model.wv.index_to_key:
        word_dict[vocab] = model.wv[vocab]

    tokenized_input = tokenized_mecab(config.input_text)
    tokenized_input_out_stopwords = out_stopwords(tokenized_input, stopword_list)

    list_vector = []
    for word in tokenized_input_out_stopwords:
        if word in word_dict.keys():
            list_vector.append(word_dict[word])

    user_vector = (np.sum(list_vector, axis=0) / len(list_vector)).tolist()

    return user_vector

''' Fine tune '''
def word2vec(config, data):
    documents = []
    for doc in data[config.terms]:
        documents.append(doc)   # doc.tolist()
    
    print('Train Word2Vec ...')
    model = Word2Vec(min_count=5,            
                     vector_size=50,
                     window=5,            
                     sg=1,    # 0: CBOW, 1: skip-gram 
                     )

    model.build_vocab(documents)
    model.train(documents,
                total_examples=len(documents),
                epochs=config.epochs)

    model.init_sims(replace=True)
    
    # Save model
    model.save(os.path.join(config.dir_path+f'/finetune/model/word2vec/{config.model_fn}'))
    
    # Load model
    model = Word2Vec.load(os.path.join(config.dir_path+f'/finetune/model/word2vec/{config.model_fn}'))
    
    return model

''' Make documents embedding '''
def documents2vec(config, raw_data, w2v_model):
    data = raw_data.copy()
    
    word_dict = {}
    for vocab in w2v_model.wv.index_to_key:
        word_dict[vocab] = w2v_model.wv[vocab]

    dict_doc_vector = {}
    for idx in data.index:
        list_vector = []

        for word in data.loc[idx][config.terms]:
            if word in word_dict.keys():
                list_vector.append(word_dict[word])

        dict_doc_vector[data.loc[idx]['sbjt_id']] = np.sum(list_vector, axis=0).tolist()

    data['vector'] = data['sbjt_id'].map(dict_doc_vector)

    return data

''' Get nearest documents '''
def get_nearest_documents(config, w2v_data, user_vector):
    similarity = {}
    
    for idx in w2v_data.index:
        if w2v_data.loc[idx]['vector'] != .0:
            sim = cosine_similarity(np.array(user_vector).reshape(1, -1), np.array([i for i in w2v_data.loc[idx]['vector']]).reshape(1, -1))
            similarity[str(w2v_data.loc[idx]['sbjt_id'])] = float(sim)

    similarity = {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    rating = [str(k) for k, v in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
    top_rank = rating[:config.rank]

    result_dict = {}
    for i in top_rank:
        result_dict[i] = str(abs(round((similarity[i] * 100), 2))) + "%"

    result_df = pd.DataFrame({'sbjt_id': similarity.keys(), 
                              'cos_sim': similarity.values()})

    result_df = pd.merge(w2v_data, result_df, on='sbjt_id', how='left').sort_values(['cos_sim'], ascending=False)
    result_df = result_df[result_df['cos_sim'] >= config.score].reset_index(drop=True)

    return result_df

def main():
    config = get_config()
    
    stopword_list = get_stopwords(config)
    raw_data = read_data(config)

    # Make Word2Vec model
    # w2v_model = word2vec(config, data)
    w2v_model = Word2Vec.load(os.path.join(config.dir_path + '/finetune/model/word2vec/tipa.w2v.refined.token.model'))
    
    # Test
    print(f"\nGet Nearest Word (Word2Vec): {config.vocab_word}")
    print(w2v_model.wv.most_similar([config.vocab_word]))
    # print(w2v_model.wv[config.vocab_word])
    
    # Get user vector
    user_vector = input2vec(config, w2v_model, stopword_list)

    # Get documents vector with Word2Vec model
    print(f'\nSearch text (Word2Vec): {config.input_text}')
    doc_vector = documents2vec(config, raw_data, w2v_model)
    
    # Get nearest documents df
    w2v_nearest_df = get_nearest_documents(config, doc_vector, user_vector)
    doc_vector_df = w2v_nearest_df[['sbjt_id', 'main_str', 'vector']]
    doc_vector_df = doc_vector_df[doc_vector_df['vector'] != .0]
    # doc_vector_df.to_parquet('doc_vector_df.parquet', engine='pyarrow', compression='gzip')
    print('')
    


if __name__ == '__main__':
    main()