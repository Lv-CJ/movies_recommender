from flask import Flask, render_template, request
from recommender import get_hybrid_recommendations, load_movies

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        age = int(request.form['age'])
        gender = request.form['gender']
        preferences = request.form.getlist('genres')
        
        recommendations = get_hybrid_recommendations(user_id, age, gender, preferences)
        movies = load_movies()

        # 生成展示信息
        movie_details = [
            {
                'title': movies.loc[movies['movieId'] == movie_id, 'title'].values[0],
                'score': round(score, 2)
            } for movie_id, score in recommendations
        ]

        return render_template('recommend.html', movie_details=movie_details)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
