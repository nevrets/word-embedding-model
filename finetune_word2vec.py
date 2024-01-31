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
    p.add_argument("--model_fn", default="word2vec_gensim.model", type=str)
    p.add_argument("--terms", default="ego_terms", type=str)             # ego_terms, refined_ego_terms
    p.add_argument("--text", default="인공지능과 자동차", type=str)
    p.add_argument("--dir_path", default="/data/nevret/word_embedding", type=str)
    
    config = p.parse_args()

    return config

''' Preprocessing '''
def get_parqet_file(path, data):
    ''' out of boundary 72184 error '''
    # data = pd.read_parquet(os.path.join(path, 'tipa_text_tokens_20220324.parquet'), engine='fastparquet')
    data = pq.read_table(os.path.join(path, data)).to_pandas()
    
    return data

def get_stopwords(stopwords_url):
    stopwords_file = list(open(stopwords_url, 'r'))
    list_stopword = []
    for i in stopwords_file:
        list_stopword.append(i[:-1])

    return list_stopword

def tokenized_mecab(data):
    result = mecab.morphs(data)
    result = ' '.join(result)
    
    return result

def out_stopwords(data, list_stopwords):
    data = data.split(' ')
    data = [token for token in data if token not in list_stopwords]

    return data

def input2vec(config, model, stopword_list):
    word_dict = {}
    for vocab in model.wv.index_to_key:
        word_dict[vocab] = model.wv[vocab]

    tokenized_input = tokenized_mecab(config.text)
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
                epochs=30)

    model.init_sims(replace=True)
    
    # Save model
    model.save(os.path.join(config.dir_path+f'/model/{config.model_fn}'))
    
    # Load model
    model = Word2Vec.load(os.path.join(config.dir_path+f'/model/{config.model_fn}'))
    
    return model

''' Make Subject Embedding '''
def documents2vec(config, data, w2v_model):
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

def get_nearest_documents(w2v_data, user_vector, rank=10, score=.5):
    similarity = {}
    
    for idx in w2v_data.index:
        if w2v_data.loc[idx]['vector'] != .0:
            sim = cosine_similarity(np.array(user_vector).reshape(1, -1), np.array([i for i in w2v_data.loc[idx]['vector']]).reshape(1, -1))
            similarity[str(w2v_data.loc[idx]['sbjt_id'])] = float(sim)

    similarity = {key: value for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    rating = [str(key) for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
    top_rank = rating[:rank]

    result_dict = {}
    for i in top_rank:
        result_dict[i] = str(abs(round((similarity[i] * 100), 2))) + "%"

    result_df = pd.DataFrame({'sbjt_id': similarity.keys(), 
                              'cos_sim': similarity.values()})

    result_df = pd.merge(w2v_data, result_df, on='sbjt_id', how='left').sort_values(['cos_sim'], ascending=False)
    # result_df = result_df[result_df['cos_sim'] >= score].reset_index(drop=True)

    return result_df

    
def remove_stopwords(col):
    stopword_list = get_stopwords(stopwords_url)
    col = [i for i in col if i not in stopword_list]

    return col


if __name__ == '__main__':
    config = get_config()
    
    stopwords_url = '/data/nevret/word_embedding/stopwords.txt'
    stopword_list = get_stopwords(stopwords_url)
    
    data = pq.read_table(os.path.join(config.dir_path + f'/tipa_text_tokens_20220328.parquet')).to_pandas().sort_values('sbjt_id').reset_index(drop=True)
    data = data.rename(columns={'terms': 'ego_terms',
                                'refined_terms': 'refined_ego_terms'})
    # data['ego_terms'] = data['ego_terms'].apply(remove_stopwords)

    # Make Word2Vec model
    w2v_model = word2vec(config, data)
    
    # Test
    print(f"\nGet Nearest Word (Word2Vec): {config.text}")
    print(w2v_model.wv.most_similar([config.text]))
    # print(w2v_model.wv[test])
    
    # Get user vector
    user_vector = input2vec(config, w2v_model, stopword_list)

    # Get documents vector with Word2Vec model
    print(f'\nSearch text (Word2Vec): {config.text}')
    w2v_data = documents2vec(config, data, w2v_model)
    
    w2v_nearest_df = get_nearest_documents(w2v_data, user_vector, rank=10, score=.5)
    sbjt_vector_df = w2v_nearest_df[['sbjt_id', 'main_str', 'vector']]
    sbjt_vector_df = sbjt_vector_df[sbjt_vector_df['vector'] != .0]
    sbjt_vector_df.to_parquet('sbjt_vector_df.parquet', engine='pyarrow', compression='gzip')
    print(w2v_nearest_df.head(10))
    
    
    print('\nCOMPLETE !!!')