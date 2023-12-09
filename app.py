import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics.pairwise

st.set_page_config(layout='wide',page_title='Book Recommendation System')

books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

ratings_with_name = ratings.merge(books,on='ISBN')

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)

popularity_df = num_rating_df.merge(avg_rating_df,on='Book-Title')

popular_df = popularity_df[popularity_df['num_ratings']>250].sort_values('avg_rating',ascending=False).head(50)

popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','num_ratings','avg_rating']]

def load_overall_analysis():
    st.title('Top 50 Books')
    popular_df

#Collaborative Filtering Based Recommender System

x = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
exp_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(exp_users)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)
similarity_score = sklearn.metrics.pairwise.cosine_similarity(pt)


def recommend(book_name):
    # index fetch
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    return [pt.index[i[0]] for i in similar_items]


st.sidebar.title('Book Recommendation System')

option = st.sidebar.selectbox('Select One',['Top 50 Books','Recommendation Based'])

if option == 'Top 50 Books':
    load_overall_analysis()

else:
    books = st.sidebar.selectbox('select book',sorted(set(pt.index)))
    btn1 = st.sidebar.button('search')
    st.title('5 similar books')
    if btn1:
        recommended_books = recommend(books)
        for book in recommended_books:
            st.write(book)



