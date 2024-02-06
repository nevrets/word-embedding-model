import pandas as pd
import numpy as np
import itertools
from functools import reduce
from copy import copy
from scipy import sparse

import os
import timeit
from gensim.models import Word2Vec
import pyarrow.parquet as pq
from konlpy.tag import Mecab
import torch
from torch import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
mecab = Mecab()


# ==============
# Preprocessing
# ==============
def get_parquet_file(path, data):
    # out of boundary 72184 error
    # data = pd.read_parquet(os.path.join(path, data), engine='fastparquet')
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

def remove_stopwords(col):
    remove_set = {'및', '할', '수', '본', '그', '한다', '된다', '때', '하지만', '이런', '이러한', '있다'}
    col = [i for i in col if i not in remove_set]

    return col


# ===========
# Modeling
# ===========
def word2vec(data, terms, model_fn, vector_size):
    corpus = []
    for term in data[terms]:
        corpus.append(term)    # doc.tolist()
    
    print('Train Word2Vec model ...')
    start_time = timeit.default_timer()

    model = Word2Vec(min_count=5,               
                     vector_size=vector_size,      
                     window=5,            
                     sg=1,    # 0: CBOW, 1: skip-gram 
                     )

    model.build_vocab(corpus)

    model.train(corpus,
                total_examples=len(corpus),
                epochs=50)

    print(f'\nGet Word2Vec model: {timeit.default_timer() - start_time:.5f} sec')

    model.init_sims(replace=True)

    model.save(model_fn)

def input2vec(user_input, model, stopword_list):
    # is_w2v = False if 'd2v' in model_fn else True
    # if is_w2v:
    #     model = Word2Vec.load(model_fn)
    # else:
    #     model = Doc2Vec.load(model_fn)

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

# word2vec을 통한 과제 vector 생성
def word2vec_subject(data, terms, w2v_model):
    word_dict = {}
    for vocab in w2v_model.wv.index_to_key:
        word_dict[vocab] = w2v_model.wv[vocab]

    # MAKE SBJT VECTOR
    dict_doc_vector = {}
    for idx in data.index:
        list_vector = []
        
        for word in data.loc[idx][terms]:
            if word in word_dict.keys():
                list_vector.append(word_dict[word])

        dict_doc_vector[data.loc[idx]['sbjt_id']] = np.sum(list_vector, axis=0).tolist()

    data['vector'] = data['sbjt_id'].map(dict_doc_vector)

    return data

# user_vector와 sbjt vector의 cosine similarity
def get_nearest_subject_word2vec(w2v_data, user_vector, rank=10, score=.5):
    similarity = {}
    
    for idx in w2v_data.index:
        if w2v_data.loc[idx]['vector'] != .0:
            sim = cosine_similarity(np.array(user_vector).reshape(1, -1), np.array([i for i in w2v_data.loc[idx]['vector']]).reshape(1, -1))
            similarity[str(w2v_data.loc[idx]['sbjt_id'])] = float(sim)

    similarity = {key:value for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    rating = [str(key) for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
    top_rank = rating[:rank]

    result_dict = {}
    for i in top_rank:
        result_dict[i] = str(abs(round((similarity[i] * 100), 2))) + "%"

    result_df = pd.DataFrame({'sbjt_id': similarity.keys(), 
                              'cos_sim': similarity.values()})

    result_df = pd.merge(w2v_data, result_df, on='sbjt_id', how='left').sort_values(['cos_sim'], ascending=False)
    result_df = result_df[['sbjt_id', 'ego_terms', 'vector', 'cos_sim']].reset_index(drop=True)
    # result_df = result_df[result_df['cos_sim'] >= score].reset_index(drop=True)

    return result_dict, result_df

# user_vector와 member vector의 cosine similarity
def get_nearest_member_word2vec(member_data, user_vector, rank=10, score=.5):
    similarity = {}
    
    for idx in member_data.index:
        if member_data.loc[idx]['mbr_vector'] != '0.0':
            sim = cosine_similarity(np.array(user_vector).reshape(1, -1), np.array([i for i in member_data.loc[idx]['mbr_vector']]).reshape(1, -1))
            similarity[str(member_data.loc[idx]['mbr_id'])] = float(sim)

    similarity = {key:value for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    rating = [str(key) for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
    top_rank = rating[:rank]

    result_dict = {}
    for i in top_rank:
        result_dict[i] = str(abs(round((similarity[i] * 100), 2))) + "%"

    result_df = pd.DataFrame({'mbr_id': similarity.keys(), 
                              'cos_sim': similarity.values()})
    
    result_df = pd.merge(member_data, result_df, on='mbr_id', how='left').sort_values(['cos_sim'], ascending=False)
    result_df = result_df.drop_duplicates('mbr_id').reset_index(drop=True)
    # result_df = result_df[result_df['cos_sim'] >= score].reset_index(drop=True)

    return result_dict, result_df


def get_total_subject_vector(PATH, target_db='tipa_mbr', get_data=False):
    if get_data:
        # 전체 과제 data
        # total_sbjt_edgelist = we_edge.get_all_edgelist(target_db=target_db)
        total_sbjt_edgelist = total_sbjt_edgelist.drop_duplicates('doc_id').sort_values('doc_id')
        total_sbjt_edgelist.to_parquet(os.path.join(PATH, 'total_sbjt_edgelist.parquet'), engine='pyarrow', compression='gzip')
    
    # total_sbjt_edgelist = pq.read_table(os.path.join(PATH, 'total_sbjt_edgelist.parquet')).to_pandas()
    total_sbjt_edgelist = pd.read_parquet(os.path.join(PATH, 'total_sbjt_edgelist.parquet'), engine='fastparquet')
    total_sbjt_edgelist = total_sbjt_edgelist.rename(columns={'doc_id': 'sbjt_id'})

    # 전체 과제 data의 vector
    # total_sbjt_vector = pq.read_table(os.path.join(PATH, 'total_sbjt_vector.parquet')).to_pandas().sort_values('sbjt_id')
    total_sbjt_vector = pd.read_parquet(os.path.join(PATH, 'total_sbjt_vector.parquet'), engine='fastparquet').sort_values('sbjt_id')
    total_sbjt_vector = total_sbjt_vector[['sbjt_id', 'vector']]

    # Merge - vector가 없는 sbjt의 경우
    # total_sbjt_edgelist > total_sbjt_vector
    total_sbjt = pd.merge(total_sbjt_edgelist, total_sbjt_vector, how='outer', on='sbjt_id')    

    ####
    for row in total_sbjt.loc[total_sbjt['vector'].isnull(), 'vector'].index:
        total_sbjt.at[row, 'vector'] = np.zeros(50, )

    return total_sbjt

def get_total_member_vector(total_sbjt, target_db='tipa_mbr'):
    # 전체 평가위원별 과제 data
    # total_mbr_edgelist = get_all_mbr_sbjt_id(target_db)
    total_mbr_edgelist = total_mbr_edgelist[['mbr_id', 'total_sbjt_list']]    # sbjt
    
    mbr_idx = list(itertools.chain.from_iterable(itertools.repeat(mbr, len(n)) for (mbr, n) in \
                    zip(total_mbr_edgelist['mbr_id'], total_mbr_edgelist['total_sbjt_list'])))
    mbr_sbjt = list(itertools.chain(*total_mbr_edgelist['total_sbjt_list']))

    total_mbr_edgelist = pd.DataFrame({'mbr_id': mbr_idx,
                                       'sbjt_id': mbr_sbjt})

    # Merge - 과제 vector & 평가위원별 과제 vector
    total_mbr_sbjt_vector = pd.merge(total_mbr_edgelist, total_sbjt, how='left', on='sbjt_id')

    start_time = timeit.default_timer()

    mbr_vector_dict = {}
    for idx in total_mbr_edgelist['mbr_id'].unique():
        eval_sbjt_vector = list(total_mbr_sbjt_vector[total_mbr_sbjt_vector['mbr_id']==idx]['vector'].values)
        
        mbr_vector_list = []
        for vector in eval_sbjt_vector:
            mbr_vector_list.append(np.array(vector))
        
        mbr_vector_dict[idx] = (np.sum(mbr_vector_list, axis=0) / len(mbr_vector_list)).tolist()

    print("Get total member vector: ", timeit.default_timer() - start_time)

    mbr_vector_df = pd.DataFrame(list(mbr_vector_dict.items()),
                                 columns=['mbr_id', 'mbr_vector'])

    total_mbr_vector = pd.merge(total_mbr_sbjt_vector, mbr_vector_df, how='left', on='mbr_id')
    total_mbr_vector = total_mbr_vector[['mbr_id', 'sbjt_id', 'mbr_vector']]

    # 전체 평가위원 data의 vector
    total_mbr = total_mbr_vector.sort_values('mbr_id').reset_index()

    total_mbr_vector.to_parquet('total_mbr_vector.parquet', engine='pyarrow', compression='gzip')

    return total_mbr_vector

def make_graph_value(one_mat, _dict, p_mode):
    labels, groups = sparse.csgraph.connected_components(one_mat)
    one_mat_deg = pd.DataFrame(one_mat.diagonal(), _dict.values())[0].to_dict()
    one_mat = sparse.triu(one_mat, k=1, format='csr')

    source, target = one_mat.nonzero()
    weight = one_mat.data            

    one_edge = pd.DataFrame({'source': source, 
                             'target': target, 
                             'weight': weight})

    one_node = pd.DataFrame({'id':      _dict.keys(),
                             'name':    _dict.values(), 
                             'cluster': groups, 
                             'linklen': one_mat_deg.values()})                           
    
    if p_mode == 'max':
        one_graph = 1
    
    one_edge['source'] = one_edge['source'].map(_dict)
    one_edge['target'] = one_edge['target'].map(_dict)
    
    return one_edge, one_node, one_graph 

def get_cosine_similarity(x1, x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5 + 1e-10)

def get_nearest(query, dataframe, metric, top_k, ascending=True):
    vector = torch.from_numpy(dataframe.loc[query].values).float()
    distances = dataframe.apply(
        lambda x: metric(vector, torch.from_numpy(x.values).float()),
        axis=1,
    )

    top_distances = distances.sort_values(ascending=ascending)[:top_k]

    print(', '.join([f'{k} ({v:.1f})' for k, v in top_distances.items()]))




if __name__ =='__main__':
    pass