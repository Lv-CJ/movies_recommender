import os
import pandas as pd
import requests
import zipfile
from io import BytesIO

def download_movielens_100k(dest_folder='data'):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print("开始下载 MovieLens 100K 数据集...")
    r = requests.get(url)
    z = zipfile.ZipFile(BytesIO(r.content))

    print("解压数据文件...")
    z.extractall(dest_folder)

    # 读取 u.data (ratings)
    ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(dest_folder, 'ml-100k', 'u.data'), sep='\t', names=ratings_cols)

    # 读取 u.item (movies)
    movies_cols = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                   'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv(os.path.join(dest_folder, 'ml-100k', 'u.item'), sep='|', names=movies_cols, encoding='latin-1')

    # 将电影类型列合并为一个字符串 genres
    genre_cols = movies_cols[5:]
    movies['genres'] = movies[genre_cols].apply(lambda row: '|'.join([genre for genre, flag in zip(genre_cols, row) if flag == 1]), axis=1)

    # 只保留 movieId, title, genres
    movies = movies[['movieId', 'title', 'genres']]

    # 读取 u.user (users)
    users_cols = ['userId', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(os.path.join(dest_folder, 'ml-100k', 'u.user'), sep='|', names=users_cols)

    # 保存成 csv
    ratings.to_csv(os.path.join(dest_folder, 'ratings.csv'), index=False)
    movies.to_csv(os.path.join(dest_folder, 'movies.csv'), index=False)
    users.to_csv(os.path.join(dest_folder, 'users.csv'), index=False)

    print("数据集已下载并保存到", dest_folder)


if __name__ == "__main__":
    download_movielens_100k()
