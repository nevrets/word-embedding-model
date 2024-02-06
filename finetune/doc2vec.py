import os
import argparse
import timeit
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

def doc2vec(config, data):
    documents = [(data[config.terms][i], i) for i in range(len(data))]    # title_keyword
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

    # Save model
    model.save(os.path.join(config.dir_path+f'/finetune/model/doc2vec/{config.model_fn}'))
    
    # Load model
    model = Doc2Vec.load(os.path.join(config.dir_path+f'/finetune/model/doc2vec/{config.model_fn}'))
    
    return model

def get_nearest_documents_doc2vec(config, data, d2v_model, stopword_list):
    def out_stopwords(data, list_stopwords):
        data = data.split(' ')
        data = [token for token in data if token not in list_stopwords]
        return data

    tokenized_input = tokenized_mecab(config.input_text)
    tokenized_input_out_stopwords = out_stopwords(tokenized_input, stopword_list)

    vectors = d2v_model.infer_vector(tokenized_input_out_stopwords)    
    print(f'\ndoc2vec: {config.input_text}')
    print(d2v_model.dv.most_similar(positive=[vectors], topn=10))

    sim_list = [i[0] for i in d2v_model.dv.most_similar([vectors], topn=config.rank) if i[1] >= config.score]
    sim_df = data.loc[sim_list].reset_index(drop=True)

    return sim_df

def main():
    config = get_config()
    
    stopword_list = get_stopwords(config)
    raw_data = read_data(config)

    # Make Doc2Vec model
    # d2v_model = doc2vec(data, terms, d2v_model_fn)
    d2v_model = Doc2Vec.load(os.path.join(config.dir_path + '/finetune/model/doc2vec/tipa.d2v.refined.token.model'))

    # Test
    print(f"\nGet Nearest Word (Doc2Vec): {config.vocab_word}")
    print(d2v_model.wv.most_similar([config.vocab_word]))
    print(d2v_model.wv[config.vocab_word])
    
    # Get documents vector with Doc2Vec model
    print(f'\nSearch Sentence (Doc2Vec): {config.input_text}')
    d2v_nearest_df = get_nearest_documents_doc2vec(config, raw_data, d2v_model, stopword_list)
    print(d2v_nearest_df.head(10))
    print('')
    


if __name__ == '__main__':
    main()
