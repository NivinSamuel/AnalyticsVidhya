import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_tags import st_tags
import torch


def load_course_data(filepath):
    df = pd.read_csv(filepath)
    return df


def embed_courses(df, model):
    course_embeddings = model.encode(
        df['Course Description'].tolist(), convert_to_tensor=True)
    return course_embeddings

# Function to search courses based on a query


def search_courses(query, model, course_embeddings, df):
    query_embedding = model.encode([query], convert_to_tensor=True)

    if torch.is_tensor(query_embedding):
        query_embedding = query_embedding.cpu().numpy()
    if torch.is_tensor(course_embeddings):
        course_embeddings = course_embeddings.cpu().numpy()

    similarity_scores = cosine_similarity(
        query_embedding, course_embeddings)[0]

    top_indices = similarity_scores.argsort()[-5:][::-1]

    results = df.iloc[top_indices].copy()
    results['Similarity Score'] = similarity_scores[top_indices]

    # Prepare the return data
    return [
        (row['Course Title'], row['Course Description'], row['Similarity Score'],
         row['Thumbnail URL'], row['Video Link'])
        for _, row in results.iterrows()
    ]

# Function to create a video thumbnail with clickable link


def video_thumbnail(title, thumbnail_url, video_link):
    if isinstance(thumbnail_url, str) and thumbnail_url.strip():
        st.markdown(
            f'<a href="{video_link}" target="_blank"><img src="{thumbnail_url}" style="width:100%; border-radius:8px;"/></a>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<a href="{video_link}" target="_blank"><img src="https://via.placeholder.com/200" style="width:100%; border-radius:8px;" /></a>', unsafe_allow_html=True)

    st.markdown(f'''
        <a href="{video_link}" target="_blank" 
        style="color:inherit; text-decoration:none; font-family: Arial, sans-serif; font-size: 18px;">
        {title}
        </a>''', unsafe_allow_html=True)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df_courses = load_course_data('Bookcsv.csv')
df_links = load_course_data('LinkPic.csv')

df = pd.merge(df_courses, df_links, on='Course Title')

course_embeddings = embed_courses(df, model)

st.set_page_config(page_title="Smart Course Search",
                   page_icon="📚", layout="wide")

st.title('Analytics Vidhya')
st.write('Find the most relevant courses of Analytics Vidhya based on your query!')

with st.sidebar:
    option = option_menu('Course Search System', ['All Courses', 'Search', 'About'],
                         icons=['book', 'search', 'info'], menu_icon='menu-button', default_index=1)

    if option == "About":
        st.header("About")
        st.write("""
            This app allows you to search for free courses on the Analytics Vidhya platform.
            It uses natural language processing to find the most relevant courses based on your search query.

            Developer: Nivin Samuel

        """)

if option == "All Courses":
    st.header("All Courses")

    num_columns = 4  # Number of videos per row
    rows = len(df) // num_columns + int(len(df) % num_columns > 0)

    for i in range(rows):
        cols = st.columns(num_columns)

        for j in range(num_columns):
            if i * num_columns + j < len(df):
                row = df.iloc[i * num_columns + j]
                with cols[j]:
                    video_thumbnail(
                        title=row['Course Title'],
                        thumbnail_url=row['Thumbnail URL'],
                        video_link=row['Video Link']
                    )

    st.markdown("---")
    st.write("© 2024 Nivin Samuel")

if option == "Search":
    st.header("Search Courses")

    course_titles = df['Course Title'].tolist()

    user_query = st_tags(
        label='Enter your search query below:',
        text='Press enter to add more',
        value=[],
        suggestions=course_titles,
        key='1'
    )

    if not user_query:
        st.subheader('Suggestions:')
        suggestions = df.sample(4)

        num_columns = 2  # 2 suggestions per row
        rows = len(suggestions) // num_columns + \
            int(len(suggestions) % num_columns > 0)

        for i in range(rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i * num_columns + j < len(suggestions):
                    row = suggestions.iloc[i * num_columns + j]
                    with cols[j]:
                        video_thumbnail(
                            title=row['Course Title'],
                            thumbnail_url=row['Thumbnail URL'],
                            video_link=row['Video Link']
                        )
                        st.markdown(
                            f"**Description:** {row['Course Description'][:150]}...")
                        st.markdown("---")

    if user_query:
        st.subheader('Search Results:')

        for query in user_query:
            results = search_courses(query, model, course_embeddings, df)

            num_columns = 2  # 2 results per row for the search results
            rows = len(results) // num_columns + \
                int(len(results) % num_columns > 0)

            for i in range(rows):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i * num_columns + j < len(results):
                        title, description, score, thumbnail_url, video_link = results[
                            i * num_columns + j]
                        with cols[j]:
                            video_thumbnail(title, thumbnail_url, video_link)
                            st.markdown(f"**Similarity Score:** {score:.2f}")
                            st.markdown("---")
