# 文件: train_svd.py
# 功能: 使用 Surprise 训练 SVD 模型

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import pickle

def train_svd_model():
    ratings = pd.read_csv('data/raw/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

if __name__ == '__main__':
    train_svd_model()

