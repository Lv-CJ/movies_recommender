# 文件: streamlit_app.py
import streamlit as st
from recommender import get_hybrid_recommendations, load_movies

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="混合电影推荐系统", layout="centered")

st.title("🎬 个性化电影推荐系统")
st.write("融合协同过滤（SVD）与内容推荐（电影类型）的混合推荐系统")

# 用户输入
user_id = st.number_input("请输入你的用户ID（如1~600）", min_value=1, step=1)
age = st.slider("选择你的年龄", 10, 80, 25)
gender = st.radio("你的性别", options=['男', '女'])

# 偏好类型
genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Thriller', 'Adventure', 'Animation']
preferences = st.multiselect("你喜欢哪些类型的电影？", genres)

# 全局变量缓存推荐结果，避免重复调用
recommendations = []
movies = None

# 🔁 主推荐按钮
if st.button("🎯 生成推荐电影", key="recommend_button_1"):
    if preferences:
        with st.spinner("正在为你生成推荐列表..."):
            recommendations = get_hybrid_recommendations(user_id, age, gender, preferences)
            movies = load_movies()
            st.success("推荐完成！为你推荐以下电影：")

            for movie_id, score in recommendations:
                title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
                st.markdown(f"**🎥 {title}**  — 推荐评分：`{round(score, 2)}`")

            # 缓存到 session state，以便图表按钮使用
            st.session_state.recommendations = recommendations
            st.session_state.movies = movies
    else:
        st.warning("请至少选择一个你感兴趣的电影类型。")

# 🔧 函数：词云图
def plot_keyword_wordcloud(preferences, movie_ids, movies):
    genre_text = " ".join(movies[movies['movieId'].isin(movie_ids)]['genres'].fillna("").values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genre_text)
    st.subheader("🎨 推荐关键词词云")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# 🔧 函数：匹配度柱状图
def plot_match_score_bar(preferences, movie_ids, movies):
    st.subheader("🔍 每部电影关键词匹配度")
    data = []
    for movie_id in movie_ids:
        genres = movies.loc[movies['movieId'] == movie_id, 'genres'].values[0]
        match_count = sum([1 for g in preferences if g.lower() in str(genres).lower()])
        title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
        data.append((title, match_count))
    df = pd.DataFrame(data, columns=["电影", "匹配度"])
    df = df.sort_values("匹配度", ascending=False)
    st.bar_chart(df.set_index("电影"))

# 📊 图表按钮
if st.button("📈 生成推荐图表", key="recommend_button_2"):
    if "recommendations" in st.session_state and "movies" in st.session_state:
        recommendations = st.session_state.recommendations
        movies = st.session_state.movies
        movie_ids = [mid for mid, _ in recommendations]

        plot_keyword_wordcloud(preferences, movie_ids, movies)
        plot_match_score_bar(preferences, movie_ids, movies)
    else:
        st.warning("请先点击上方按钮生成推荐结果。")
