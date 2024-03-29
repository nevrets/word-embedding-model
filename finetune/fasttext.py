import os
import timeit
import argparse
import pandas as pd
import numpy as np

from copy import copy
from gensim.models.fasttext import FastText

import pyarrow.parquet as pq
from konlpy.tag import Mecab
# from torch import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import *

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
    stopwords_file = list(open(os.path.join(config.dir_path, 'finetune/utils/stopwords.txt'), 'r'))
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

''' Modeling '''
def fasttext(config, data):
    documents = []
    for doc in data[config.terms]:
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

    # Save model
    model.save(os.path.join(config.dir_path+f'/finetune/model/fasttext/{config.model_fn}'))
    
    # Load model
    model = FastText.load(os.path.join(config.dir_path+f'/finetune/model/fasttext/{config.model_fn}'))
    
    return model
    

def documents2vec_fasttext(config, raw_data, ft_model):
    data = raw_data.copy()
    
    word_dict = {}
    for vocab in ft_model.wv.index_to_key:
        word_dict[vocab] = ft_model.wv[vocab]

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
def get_nearest_documents_fasttext(config, ft_data, user_vector):
    similarity = {}
    
    for idx in ft_data.index:
        if ft_data.loc[idx]['vector'] != .0:
            sim = cosine_similarity(np.array(user_vector).reshape(1, -1), np.array([i for i in ft_data.loc[idx]['vector']]).reshape(1, -1))
            similarity[str(ft_data.loc[idx]['sbjt_id'])] = float(sim)

    similarity = {key: value for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    rating = [str(key) for key, value in sorted(similarity.items(), key=lambda item: item[1], reverse=True)]
    top_rank = rating[:config.rank]

    result_dict = {}
    for i in top_rank:
        result_dict[i] = str(abs(round((similarity[i] * 100), 2))) + "%"

    result_df = pd.DataFrame({'sbjt_id': similarity.keys(), 'cos_sim': similarity.values()})
    result_df = pd.merge(ft_data, result_df, on='sbjt_id', how='left').sort_values(['cos_sim'], ascending=False)
    result_df = result_df[result_df['cos_sim'] >= config.score].reset_index(drop=True)

    return result_df

def main():
    config = get_config()
    
    stopword_list = get_stopwords(config)
    raw_data = read_data(config)

    # Make FastText model
    # fasttext_model = fasttext(config, data)
    fasttext_model = FastText.load(os.path.join(config.dir_path + '/finetune/model/fasttext/tipa.ft.refined.token.model'))
    
    # Test
    print(f"\nGet Nearest Word (FastText): {config.vocab_word}")
    print(fasttext_model.wv.most_similar([config.vocab_word]))
    # print(fasttext_model.wv[config.vocab_word])

    # Get user vector
    user_vector = input2vec(config, fasttext_model, stopword_list)

    # Get documents vector with Word2Vec model
    print(f'\nSearch text (Word2Vec): {config.input_text}')
    doc_vector = documents2vec_fasttext(config, raw_data, fasttext_model)
    
    # Get nearest documents df
    fasttext_nearest_df = get_nearest_documents_fasttext(config, doc_vector, user_vector)
    print(fasttext_nearest_df.head(10))
    print('')
    


if __name__ == '__main__':
    main()

