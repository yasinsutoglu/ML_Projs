############################################
# User-Based Collaborative Filtering
#############################################

# Step 1: Preparing the Data Set
# Step 2: Determining the Movies Watched by the User to Make a Recommendation
# Step 3: Accessing the Data and IDs of Other Users Watching the Same Movies
# Step 4: Determining the Users with the Most Similar Behavior to the User to Make a Recommendation
# Step 5: Calculating the Weighted Average Recommendation Score
# Step 6: Functionalization of the Work

#############################################
# Step 1: Preparing the Data Set

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    
    comment_counts = pd.DataFrame(df["title"].value_counts())    
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    
    return user_movie_df

user_movie_df = create_user_movie_df()

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

#############################################
# Step 2: Determining the Movies Watched by the User to Make a Recommendation

print(random_user)
print(10 * "--")
print(user_movie_df)

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

print(10 * "--")
print(len(movies_watched))

#############################################
# Step 3: Accessing the Data and IDs of Other Users Watching the Same Movies


movies_watched_df = user_movie_df[movies_watched]

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# user_movie_count[user_movie_count["movie_count"] == 33].count()

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# ---more dynamic way to apply---
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Step 4: Determining the Users with the Most Similar Behavior to the User to Make a Recommendation

# For this case, I will perform 3 steps:
# 1. I will bring together the data of X_named_user and other users.
# 2. I will create the correlation df.
# 3. I will find the most similar users (Top Users)

final_df = pd.concat([ movies_watched_df[movies_watched_df.index.isin(users_same_movies)] , random_user_df[movies_watched] ])

final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()


top_users = corr_df[ (corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65) ][ ["user_id_2", "corr"] ].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Step 5: Calculating the Weighted Average Recommendation Score

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# print(recommendation_df[["movieId"]].nunique())

# recommendation_df[recommendation_df["weighted_rating"] > 3.7]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.7].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('datasets/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Step 6: Functionalization All of the Work
#############################################

# def create_user_movie_df():
#     import pandas as pd
    
#     movie = pd.read_csv('datasets/movie.csv')
#     rating = pd.read_csv('datasets/rating.csv')
#     df = movie.merge(rating, how="left", on="movieId")
    
#     comment_counts = pd.DataFrame(df["title"].value_counts())
#     rare_movies = comment_counts[comment_counts["title"] <= 1000].index
#     common_movies = df[~df["title"].isin(rare_movies)]
#     user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    
#     return user_movie_df

# user_movie_df = create_user_movie_df()

# ------------------------
# def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.7):
#     import pandas as pd
    
#     random_user_df = user_movie_df[user_movie_df.index == random_user]
    
#     movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
#     movies_watched_df = user_movie_df[movies_watched]
#     user_movie_count = movies_watched_df.T.notnull().sum()
#     user_movie_count = user_movie_count.reset_index()
#     user_movie_count.columns = ["userId", "movie_count"]
    
#     perc = len(movies_watched) * ratio / 100
#     users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#     final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
#                           random_user_df[movies_watched]])

#     corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
#     corr_df = pd.DataFrame(corr_df, columns=["corr"])
#     corr_df.index.names = ['user_id_1', 'user_id_2']
#     corr_df = corr_df.reset_index()

#     top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
#         ["user_id_2", "corr"]].reset_index(drop=True)

#     top_users = top_users.sort_values(by='corr', ascending=False)
#     top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
#     rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#     top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
#     top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

#     recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
#     recommendation_df = recommendation_df.reset_index()

#     movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
#     movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    
#     return movies_to_be_recommend.merge(movie[["movieId", "title"]])
# ------------------------
# random_user = int(pd.Series(user_movie_df.index).sample(1).values)
# user_based_recommender(random_user, user_movie_df) # fucntion calling

