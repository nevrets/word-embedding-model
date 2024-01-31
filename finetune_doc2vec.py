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
mecab = Mecab()


def get_config():
    p = argparse.ArgumentParser(description="Set arguments.")

    p.add_argument("--seed", default="42", type=int)
    p.add_argument("--file_name", default="acc_final_preproc", type=str)
    p.add_argument("--terms", default="ego_terms", type=str)
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
    # list_stopwords = list_stopwords
    data = [token for token in data if token not in list_stopwords]

    return data

def input2vec(user_input, model, stopword_list):
    word_dict = {}
    for vocab in model.wv.index_to_key:
        word_dict[vocab] = model.wv[vocab]

    tokenized_input = tokenized_mecab(user_input)
    tokenized_input_out_stopwords = out_stopwords(tokenized_input, stopword_list)

    list_vector = []
    for word in tokenized_input_out_stopwords:
        if word in word_dict.keys():
            list_vector.append(word_dict[word])

    user_vector = (np.sum(list_vector, axis=0) / len(list_vector)).tolist()

    return user_vector

''' Modeling '''
def word2vec(data, terms, model_fn):
    documents = []
    for doc in data[terms]:
        documents.append(doc)   # doc.tolist()
    
    print('Train Word2Vec model ...')
    start_time = timeit.default_timer()

    model = Word2Vec(min_count=5,               
                     vector_size=50,      
                     window=5,            
                     sg=1,    # 0: CBOW, 1: skip-gram 
                     )

    model.build_vocab(documents)

    model.train(documents,
                total_examples=len(documents),
                epochs=30)

    print(f'\nEnd: {timeit.default_timer() - start_time:.5f} sec')

    model.init_sims(replace=True)

    model.save(model_fn)

''' Make Subject Embedding '''
def word2vec_documents(data, terms, w2v_model):
    word_dict = {}
    for vocab in w2v_model.wv.index_to_key:
        word_dict[vocab] = w2v_model.wv[vocab]

    dict_doc_vector = {}
    for idx in data.index:
        list_vector = []
        
        for word in data.loc[idx][terms]:
            if word in word_dict.keys():
                list_vector.append(word_dict[word])

        dict_doc_vector[data.loc[idx]['sbjt_id']] = np.sum(list_vector, axis=0).tolist()

    data['vector'] = data['sbjt_id'].map(dict_doc_vector)

    return data

def get_nearest_documents_word2vec(w2v_data, user_vector, rank=10, score=.5):
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


def doc2vec(data, terms, model_fn):
    documents = [(data[terms][i], i) for i in range(len(data))]    # title_keyword
    tagged_documents = [TaggedDocument(d, [c]) for d, c in documents]

    print('\nTrain Doc2Vec Model ...')
    start_time = timeit.default_timer()

    model = Doc2Vec(vector_size=50,
                    window=5,
                    min_count=5,
                    )

    model.build_vocab(tagged_documents)

    model.train(tagged_documents,
                total_examples=model.corpus_count,
                epochs=30,
                )

    print(f'\nEnd: {timeit.default_timer() - start_time:.5f} sec')
    
    model.init_sims(replace=True)

    model.save(model_fn)

def get_nearest_documents_doc2vec(data, d2v_model, sentence, rank=10, score=.5):
    tokenized_input = tokenized_mecab(sentence)
    tokenized_input_out_stopwords = out_stopwords(tokenized_input, stopword_list)

    vectors = d2v_model.infer_vector(tokenized_input_out_stopwords)    
    print(f'\ndoc2vec: {text}')
    print(d2v_model.dv.most_similar(positive=[vectors], topn=10))

    sim_list = [i[0] for i in d2v_model.dv.most_similar([vectors], topn=rank) if i[1] >= score]
    sim_df = data.loc[sim_list].reset_index(drop=True)

    return sim_df


def fasttext(data, terms, model_fn):
    documents = []
    for doc in data[terms]:
        documents.append(doc)
    
    print('Train FastText model ...')
    start_time = timeit.default_timer()

    model = FastText(min_count=5,          
                     vector_size=100,      
                     window=5,      
                     sg=1,    # 0: CBOW,  1: Skip-gram
                     )

    model.build_vocab(documents)

    model.train(documents,
                total_examples=len(documents),
                epochs=50)

    model.init_sims(replace=True)

    print(f'\nEnd: {timeit.default_timer() - start_time:.5f} sec')

    model.save(model_fn)

def fasttext_documents(data, terms, ft_model):
    word_dict = {}
    for vocab in ft_model.wv.index_to_key:
        word_dict[vocab] = ft_model.wv[vocab]

    dict_doc_vector = {}
    for idx in data.index:
        list_vector = []
        
        for word in data.loc[idx][terms]:
            if word in word_dict.keys():
                list_vector.append(word_dict[word])

        dict_doc_vector[data.loc[idx]['sbjt_id']] = np.sum(list_vector, axis=0).tolist()

    data['vector'] = data['sbjt_id'].map(dict_doc_vector)

    return data

def get_nearest_documents_fasttext(ft_data, user_vector, rank=10, score=.5):
    similarity = {}
    
    for idx in ft_data.index:
        if ft_data.loc[idx]['vector'] != .0:
            sim = cosine_similarity(np.array(user_vector).reshape(1, -1), np.array([i for i in ft_data.loc[idx]['vector']]).reshape(1, -1))
            similarity[str(ft_data.loc[idx]['sbjt_id'])] = float(sim)

    similarity = {key: value for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    rating = [str(key) for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
    top_rank = rating[:rank]

    result_dict = {}
    for i in top_rank:
        result_dict[i] = str(abs(round((similarity[i] * 100), 2))) + "%"

    result_df = pd.DataFrame({'sbjt_id': similarity.keys(), 'cos_sim': similarity.values()})
    result_df = pd.merge(ft_data, result_df, on='sbjt_id', how='left').sort_values(['cos_sim'], ascending=False)
    result_df = result_df[result_df['cos_sim'] >= score].reset_index(drop=True)

    return result_df
    
def remove_stopwords(col):
    remove_set = {'및', '할', '수', '본', '그'}
    col = [i for i in col if i not in remove_set]

    return col


if __name__ == '__main__':
    config = get_config()
    
    terms = 'ego_terms'    # terms
    test = '인공지능'
    text = '인공지능과 자동차'

    w2v_model_fn =  'tipa_model/tipa.w2v.refined.token.model'
    d2v_model_fn = 'tipa_model/tipa.d2v.refined.token.model'
    ft_model_fn = 'tipa_model/tipa.ft.refined.token.model'
    stopwords_url = 'tipa_model/stopwords.txt'

    data = pq.read_table(os.path.join(config.dir_path + f'/tipa_text_tokens_20220328.parquet')).to_pandas().sort_values('sbjt_id').reset_index(drop=True)
    tmp = pq.read_table(os.path.join(config.dir_path + f'/sbjt_vector_df.parquet')).to_pandas().sort_values('sbjt_id').reset_index(drop=True)
    data = data.rename(columns={'terms': 'ego_terms',
                                'refined_terms': 'refined_ego_terms'})
  
    stopword_list = get_stopwords(stopwords_url)
    
    data['ego_terms'] = data['ego_terms'].apply(remove_stopwords)

    # ====================================================================

    # Make Word2Vec model
    word2vec(data, terms, w2v_model_fn)
    
    # Load Word2Vec model
    w2v_model = Word2Vec.load(w2v_model_fn)
    
    # Test
    print(f"\nGet Nearest Word (Word2Vec): {test}")
    # print(w2v_model.wv[test])
    print(w2v_model.wv.most_similar([test]))
    
    # Get user vector
    user_vector_w2v = input2vec(text, w2v_model, stopword_list)

    # Get documents vector with Word2Vec model
    print(f'\nSearch text (Word2Vec): {text}')
    w2v_data = word2vec_documents(data, terms, w2v_model)
    w2v_nearest_df = get_nearest_documents_word2vec(w2v_data, user_vector_w2v, rank=10, score=.5)
    sbjt_vector_df = w2v_nearest_df[['sbjt_id', 'main_str', 'vector']]
    sbjt_vector_df = sbjt_vector_df[sbjt_vector_df['vector'] != .0]
    sbjt_vector_df.to_parquet('sbjt_vector_df.parquet', engine='pyarrow', compression='gzip')
    print(w2v_nearest_df.head(10))
    
    # ====================================================================

    # Make Doc2Vec model
    # doc2vec(data, terms, d2v_model_fn)

    # Load Doc2Vec model
    d2v_model = Doc2Vec.load(d2v_model_fn)

    # Get documents vector with Doc2Vec model
    print(f'\nSearch Sentence (Doc2Vec): {text}')
    d2v_nearest_df = get_nearest_documents_doc2vec(data, d2v_model, text, rank=10, score=.5)
    print(d2v_nearest_df.head(10))
    
    # ====================================================================
    
    # Make FastText model
    # fasttext(data, terms, ft_model_fn)

    # Load FastText model
    ft_model = FastText.load(ft_model_fn)

    # Test
    print(f"\nGet Nearest Word (FastText): {test}")
    # print(ft_model.wv[test])
    print(ft_model.wv.most_similar(test))

    # Get user vector
    user_vector_ft = input2vec(text, ft_model, stopword_list)

    # Get documents vector with Word2Vec model
    print(f'\nSearch text: {text}')
    ft_data = fasttext_documents(data, terms, ft_model)
    ft_nearest_df = get_nearest_documents_fasttext(ft_data, user_vector_ft, rank=10, score=.5)
    print(ft_nearest_df.head(10))
    
    print('\nCOMPLETE !!!')