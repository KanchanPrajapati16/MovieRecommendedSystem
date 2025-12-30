import streamlit as st
import pickle
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #064e3b, #022c22);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ecfdf5;
}
.subtitle {
    color: #a7f3d0;
    font-size: 18px;
    margin-bottom: 30px;
}
.highlight-box {
    background: rgba(6, 78, 59, 0.85);
    border: 2px solid #34d399;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 0 30px rgba(52, 211, 153, 0.6);
    margin-top: 25px;
}
.card {
    background: #022c22;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    transition: 0.3s;
    border: 1px solid #34d399;
}
.card:hover {
    transform: scale(1.07);
    box-shadow: 0px 10px 30px rgba(52, 211, 153, 0.8);
}
.fake-poster {
    height: 220px;
    border-radius: 12px;
    background: linear-gradient(
        120deg,
        #065f46 30%,
        #34d399 38%,
        #065f46 48%
    );
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    margin-bottom: 15px;
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.stButton>button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    font-size: 16px;
    padding: 12px 30px;
    border-radius: 14px;
    border: none;
    font-weight: bold;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #16a34a, #15803d);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

#  LOAD The DATA 
movies = pickle.load(open("movies.pkl", "rb"))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart movie recommendations</div>', unsafe_allow_html=True)

selected_movie = st.selectbox(
    " Select a movie",
    movies['title'].values
)
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

if st.button("✨ Recommend Movies"):
    with st.spinner("Finding best recommendations..."):
        time.sleep(1.5)
        recommendations = recommend(selected_movie)

    st.markdown("""
    <div class="highlight-box">
        <h2 style="color:#ecfdf5; text-align:center;">🍿 Recommended Movies</h2>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for col, movie_name in zip(cols, recommendations):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="fake-poster"></div>
                <h4 style="color:#ecfdf5;">{movie_name}</h4>
                <p style="color:#a7f3d0;">Recommended for you</p>
            </div>
            """, unsafe_allow_html=True)
