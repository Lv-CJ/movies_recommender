import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pickle
import os

def train_rf_classifier():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    users = pd.read_csv('data/users.csv')

    ratings['like'] = ratings['rating'].apply(lambda x: 1 if x >= 4 else 0)
    data = ratings.merge(movies, on='movieId').merge(users, on='userId')

    # 电影类别 One-Hot
    genres = data['genres'].str.get_dummies(sep='|')
    data['gender_encoded'] = data['gender'].map({'M': 0, 'F': 1})

    X = pd.concat([data[['age', 'gender_encoded']], genres], axis=1)
    y = data['like']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("随机森林 F1-score:", f1_score(y_test, y_pred))
    print("随机森林 AUC-ROC:", roc_auc_score(y_test, y_proba))

    os.makedirs('models', exist_ok=True)
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

if __name__ == "__main__":
    train_rf_classifier()
