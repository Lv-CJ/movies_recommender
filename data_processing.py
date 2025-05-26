# 文件: data_processing.py
# 功能: 加载并预处理 MovieLens 数据

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data():
    ratings = pd.read_csv('data/raw/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    rating_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return rating_matrix