#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

pd.set_option('display.max_columns', None)

# Step 1: Preparing the Data Set
# Step 2: Modeling
# Step 3: Model Tuning
# Step 4: Final Model and Prediction

#############################
# Step 1: Preparing the Data Set

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")

print(df.head())

movie_ids = [130219, 356, 4422, 541]
# movies = ["The Dark Knight (2011)",
#           "Cries and Whispers (Viskningar och rop) (1972)",
#           "Forrest Gump (1994)",
#           "Blade Runner (1982)"]

# I created a sample data set from the available movie list
sample_df = df[df.movieId.isin(movie_ids)]
print(sample_df.head())
print(20 * "*")
print(sample_df.shape)

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")
print(20 * "*")
print(user_movie_df.shape) #(76918,4) => 76918 : users; 4 : films

reader = Reader(rating_scale=(1, 5)) # I created the scale range I will determine

# I graded the data on a scale
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Step 2: Modeling

# Let's remember the concept that in ML problems, I should separate the data sets into trainset and testset,
# build the model on a training set, and then test it on the test set that has not seen the model before.

trainset, testset = train_test_split(data, test_size=.25) # %75 train_set, %25 test_set 
svd_model = SVD() # matrix fact. class object
svd_model.fit(trainset) # model fitting/establishing. I found p and q weights via trainset
predictions = svd_model.test(testset) # dataset blank cells' estimations are made.

accuracy.rmse(predictions) # for minimization process
# here is the expected average error to be made.

# I made a prediction for the specific user
svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True) # estimated rating=4.16
sample_df[sample_df["userId"] == 1] # real rating value = 4.0

##############################
# Step 3: Model Fine-Tuning (hyperparameter optimization)

# It is the process of optimizing the base model. In other words, It is increasing model prediction performance.
# The issue is how to optimize the parameters that are external to the model/open to user intervention/hyperparameter.

# epoch number => iteration number (number of iteration of the Stochastic Gradient Descent procedure)
# lr_all => learning rate 
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}
# hiperparameters are the Args included in the SVD() function detail.

# 'rmse', 'mae'(mean absolute error) => error evaluation metrics
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3, # 3 phase cross validation process : Divide the dataset into 3 and build a model with 2 parts
                  # Test with 1 part. Repeat this in combinatoric way! Take the average of these test runs!
                  n_jobs=-1, # Use processors at full performance.
                  joblib_verbose=True) # reporting during process

gs.fit(data)

print(gs.best_score['rmse']) # 0.93.. 
print(gs.best_params['rmse'])

##############################
# Step 4: Final Model and Prediction

dir(svd_model)
print(svd_model.n_epochs)

svd_model = SVD(**gs.best_params['rmse']) 

data = data.build_full_trainset() # The entire data set became a full trainset.
svd_model.fit(data)

# I created predictions for specific movie ("Blade Runner")
svd_model.predict(uid=1.0, iid=541, verbose=True) # est = 4.20.. ; realValue => 4.0 






