import numpy as np
import pandas as pd
import itertools

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# word2vec을 통한 과제 embedding 생성
def word2vec_subject(data, w2v_model):
    word_dict = {}
    for vocab in w2v_model.wv.index_to_key:
        word_dict[vocab] = w2v_model.wv[vocab]

    # MAKE SBJT VECTOR
    doc_vector_dict = {}
    for idx in data.index:
        list_vector = []
        
        for word in data.loc[idx]['term']:
            if word in word_dict.keys():
                list_vector.append(word_dict[word])

        doc_vector_dict[data.loc[idx]['doc_id']] = np.sum(list_vector, axis=0).tolist()

    data['vector'] = data['doc_id'].map(doc_vector_dict)

    return data

# word2vec을 통한 평가위원 embedding 생성
def word2vec_member(mbr_list, sbjt_vector):
    mbr_idx = list(itertools.chain.from_iterable(itertools.repeat(mbr, len(n)) for (mbr, n) in \
                    zip(mbr_list['mbr_id'], mbr_list['total_sbjt_list'])))
    mbr_sbjt = list(itertools.chain(*mbr_list['total_sbjt_list']))

    mbr_list = pd.DataFrame({'mbr_id': mbr_idx,
                             'doc_id': mbr_sbjt})

    # Merge - 과제 embedding & 평가위원별 과제 embedding
    mbr_sbjt_vector = pd.merge(mbr_list, sbjt_vector, how='left', on='doc_id')

    # 평가위원 embedding 생성
    mbr_vector_dict = {}
    for idx in mbr_list['mbr_id'].unique():
        eval_sbjt_vector = list(mbr_sbjt_vector[mbr_sbjt_vector['mbr_id']==idx]['vector'].values)
        
        mbr_vector_list = []
        for vector in eval_sbjt_vector:
            mbr_vector_list.append(np.array(vector))
        
        mbr_vector_dict[idx] = (np.sum(mbr_vector_list, axis=0) / len(mbr_vector_list)).tolist()

    mbr_vector_df = pd.DataFrame(list(mbr_vector_dict.items()),
                                columns=['mbr_id', 'mbr_vector'])

    # mbr vector merge했을시 예외처리 추가
    mbr_vector = None
    try:
        mbr_vector = pd.merge(mbr_sbjt_vector, mbr_vector_df, how='left', on='mbr_id')
        mbr_vector = mbr_vector[['mbr_id', 'mbr_vector']].drop_duplicates(['mbr_id'])
        
    except ValueError:
        mbr_vector = mbr_vector_df

        # mbr_sbjt_vector['mbr_id'] = mbr_sbjt_vector.astype(object)
        # mbr_vector = mbr_sbjt_vector.reindex_axis(mbr_sbjt_vector.columns.union(mbr_vector_df.columns), axis=1)

    # 전체 평가위원의 embedding
    mbr_vector = mbr_vector.sort_values('mbr_id').reset_index(drop=True)

    return mbr_vector


# word2vec을 이용한 분과 embedding 계산 후 분과와 평가위원의 cosine similarity 계산
def get_cosine_similarity(grp_key, w2v_sbjt_vector, w2v_mbr_vector):
    # 분과 embedding 계산
    grp_vector = []
    for vector in w2v_sbjt_vector['vector']:
        grp_vector.append(np.array(vector))

    w2v_grp_vector = (np.sum(grp_vector, axis=0) / len(grp_vector)).tolist()

    # Cosine similarity 계산
    cos_sim = []
    for vector in w2v_mbr_vector['mbr_vector']:
        similarities = cosine_similarity(np.array(w2v_grp_vector).reshape(1, -1), np.array(vector).reshape(1, -1))
        cos_sim.append(similarities.tolist())

    cos_sim = pd.DataFrame(list(itertools.chain(*list(itertools.chain(*cos_sim)))), columns={'similarity'})

    # 최종 result (dataframe 형태)
    df_result = pd.concat([w2v_mbr_vector, cos_sim], axis=1)
    df_result['grp_key'] = grp_key
    df_result = df_result[['grp_key', 'mbr_id', 'similarity']]

    return df_result    

# 전체 process
def total_process(result, model_path):
    # Word embedding model
    w2v_model = Word2Vec.load(model_path)

    # 과제 embedding
    w2v_sbjt_vector = word2vec_subject(result['sbjt_ego_edgelist'], w2v_model)

    # 평가위원 embedding
    w2v_mbr_vector = word2vec_member(result['mbr_sbjt_df'], w2v_sbjt_vector)

    # 분과 embedding 계산 후 분과와 평가위원의 코사인 유사도 계산
    result_df = get_cosine_similarity(result['grp_key'], w2v_sbjt_vector, w2v_mbr_vector)

    return result_df


if __name__ == '__main__':
    pass