# 文件: svd_predictor.py
# 功能: 基于 SVD 进行推荐

import pickle
import pandas as pd

def get_svd_recommendations(user_id, top_n=5):
    with open('svd_model.pkl', 'rb') as f:
        model = pickle.load(f)

    ratings = pd.read_csv('data/raw/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    all_movies = ratings['movie_id'].unique()
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    unrated_movies = [m for m in all_movies if m not in rated_movies]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in unrated_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_n]
