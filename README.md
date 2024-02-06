# 평가위원 기술용어 유사도 측정

### Introduction
평가위원의 기술용어 데이터를 정의하고 사용자가 검색한 단어 및 문장과 가장 유사한 연구자를 찾기 위함.

### Structure
```
├── word-embedding-model
│   ├── api
│   │   ├── calc_word2vec_cos_sim.py
│   │   └── embedding.py
│   ├── finetune.py
│   │   ├── utils
│   │   │   ├── utils.py
│   │   │   └── stopwords.py
│   │   ├── doc2vec.py
│   │   ├── fasttext.py
│   │   └── word2vec.py
│   ├── README.md
│   └── requirements.txt
```

### Process
- Word2vec 모델을 사용하여 과제의 단어를 임베딩으로 만든 뒤, 이를 토대로 평가위원이 평가한 과제에 따라 평가위원 임베딩을 만듦 
<br>
- 사용자의 검색어를 인풋으로 받아서 코사인 유사도를 통해 검색어와 유사한 평가위원을 추천

<br>

1.	과제 임베딩 생성 <br>

    - 과제 텍스트의 edgelist term을 이용해 과제 정보를 단어화하고 Word2Vec을 사용해 임베딩으로 만든다. <br>
    - 과제 텍스트를 mecab으로 tokenizing하는 방법도 있지만 전반적으로 용어 자체가 mecab에 학습되지 않은 것이 많이 때문에 사용자 사전에 추가하는 작업이 필수적 <br>
    - 추출된 단어들의 임베딩을 word2vec으로 구하고 이를 전부 다 합쳐서 과제 임베딩으로 만든다. <br>


2.	평가위원 임베딩 생성

    - 평가위원이 평가한 과제 임베딩을 평가위원별로 정리한다.
    - 평가위원별 과제 임베딩을 가중평균으로 계산하여 평가위원 임베딩으로 만든다.


3.	사용자 임베딩 생성 및 평가위원 추천

    - 사용자의 검색어 임베딩과 평가위원 임베딩 사이의 cosine similarity를 구한다.
    - 유사도가 높은 순으로 정리해서 연구자를 추천한다.

<br>

### Reference
https://github.com/hw79chopin/National-assembly-member-recommder.git