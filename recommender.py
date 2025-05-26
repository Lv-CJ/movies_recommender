import pandas as pd
import pickle
from surprise import Dataset, Reader

# 加载模型和数据
def load_svd_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)

def load_rf_model():
    with open('rf_model.pkl', 'rb') as f:
        return pickle.load(f)

def load_movies():
    return pd.read_csv('data/movies.csv')

def load_users():
    return pd.read_csv('data/users.csv')

def get_svd_score(svd_model, user_id, movie_id):
    return svd_model.predict(user_id, movie_id).est

def get_rf_score(rf_model, user_features):
    return rf_model.predict_proba([user_features])[0][1]

def get_hybrid_recommendations(user_id, age, gender, preferences, top_n=10):
    movies = load_movies()
    users = load_users()
    svd_model = load_svd_model()
    rf_model = load_rf_model()

    gender_encoded = 0 if gender == '男' else 1

    # 筛选符合偏好的电影
    candidate_movies = movies[movies['genres'].apply(lambda x: any(g in x for g in preferences))]

    recommendations = []
    for _, movie in candidate_movies.iterrows():
        # SVD 评分预测
        svd_score = get_svd_score(svd_model, user_id, movie['movieId'])

        # 构造用户电影特征给RF模型
        movie_genres = movie['genres'].split('|')
        genres_list = movies['genres'].str.get_dummies(sep='|').columns.tolist()
        genre_features = [1 if g in movie_genres else 0 for g in genres_list]

        user_features = [age, gender_encoded] + genre_features

        rf_score = get_rf_score(rf_model, user_features)

        # 加权融合评分（可调整权重）
        hybrid_score = 0.6 * svd_score + 0.4 * rf_score

        recommendations.append((movie['movieId'], hybrid_score))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations
