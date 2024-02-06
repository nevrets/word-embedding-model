import os, re
import numpy as np

import torch

def get_logger(filename='./train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
    
    logger.setLevel(INFO)

    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger

def seed_everything(seed:int = 1004):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  

def clean_data(text):
    def load_stopwords():
        with open('/data/nevret/word_embedding/stopwords.txt', 'r') as f:
            stopwords = f.readlines()
        return stopwords[0].split(',')

    stopwords = load_stopwords()
    # words = ''.join(text.tolist())

    # 불용어 제거
    text = [x for x in text.split(' ') if x not in stopwords and len(x) > 1]

    # join 후 공백 제거
    return ' '.join(text).strip()