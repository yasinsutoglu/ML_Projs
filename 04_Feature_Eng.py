#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# !pip install missingno
import missingno as msno

from datetime import date

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
print(df.head())
print(10 * "*")


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df2 = load()
df2.head()

#############################################
# 1. OUTLIERS

#############################################
# 1.1 Capturing Outliers

# Capturing Outliers with Chart Technique

sns.boxplot(x=df2["Age"])
plt.show()

#-----------------------------------
# How to Catch Outliers?
#-----------------------------------
q1 = df2["Age"].quantile(0.25)
q3 = df2["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# df2[(df2["Age"] < low) | (df2["Age"] > up)]

print(10 * "*")
print(df2[(df2["Age"] < low) | (df2["Age"] > up)].index)
# Index([33, 54, 96, 116, 280, 456, 493, 630, 672, 745, 851], dtype='int64')

#-----------------------------------
# Are There Any Outliers or No?
#-----------------------------------
df2[(df2["Age"] < low) | (df2["Age"] > up)].any(axis=None)
df2[~((df2["Age"] < low) | (df2["Age"] > up))].shape # those who are not Outliers
df2[(df2["Age"] < low)].any(axis=None)

# 1. threshold value is set
# 2. reaching the outliers.
# 3. quickly ask if there were any outliers.

###################
# Functionalization of Above Outliers Process

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    
    interquantile_range = quartile3 - quartile1
    
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    
    return low_limit, up_limit
    
print(10 * "*")
print(outlier_thresholds(df2, "Age")) # (-6.6875, 64.8125)
print(outlier_thresholds(df2, "Fare")) # (-26.724, 65.6344)

low, up = outlier_thresholds(df2, "Fare")

print(10 * "*")
print(df2[(df2["Fare"] < low) | (df2["Fare"] > up)].head())
print(df2[(df2["Fare"] < low) | (df2["Fare"] > up)].index)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

print(10 * "*")
print(check_outlier(df2, "Age"))
print(check_outlier(df2, "Fare"))

#---------------------------------
# grab_col_names => check_outlier(df2, "col_names") --> to automate col_names here.
#---------------------------------

dff = load_application_train()
print(10 * "*")
print(dff.head()) # There are 122 variables here, which ones are numeric and which ones are not, 
# I need to parse them and give them to the check_outlier() function.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and cardinal variables that appear to be categorical in the data set.
    Note: Categorical variables with numerical view are also included.

    Parameters
    ------
        dataframe: dataframe
                Dataframe from which variable names are to be taken
        cat_th: int, optional
                Class threshold value for variables that are numeric but categorical
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                List of cardinal variables with categorical view

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        Returned 3 list => cat_cols + num_cols + cat_but_car = total number of variables
        "num_but_cat" contains "cat_cols"
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    
    cat_cols = cat_cols + num_but_cat
    # Let's get to the final categorical list:
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols => I took the ones whose type is different from object
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    return cat_cols, num_cols, cat_but_car

# Function calling:
cat_cols, num_cols, cat_but_car = grab_col_names(df2)

num_cols = [col for col in num_cols if col not in "PassengerId"]
for col in num_cols:
    print(col, check_outlier(df2, col))

# Function calling:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
for col in num_cols:
    print(col, check_outlier(df, col))

###################
# Accessing the Outliers 

def grab_outliers(dataframe, col_name, index=False):
    
    low, up = outlier_thresholds(dataframe, col_name)

    # dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] =>  the number of observations
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        
        return outlier_index

# Function calling:
grab_outliers(df2, "Age")

print(grab_outliers(df2, "Age", True))

age_index = grab_outliers(df2, "Age", True)

# outlier_thresholds(df2, "Age")
# check_outlier(df2, "Age")
# grab_outliers(df2, "Age", True)

# NOTE:  Because many of the tree methods are sensitive to outliers, outliers are must be detected and excluded from dataframe.

#############################################
# Methods for Solving the Outlier Problem

#------------------------
# 1. Dropping Method
#------------------------

low, up = outlier_thresholds(df2, "Fare")
print(df2.shape) # (891, 12)

# Number of observations excluding outliers
print(df2[~((df2["Fare"] < low) | (df2["Fare"] > up))].shape) 
# (775, 12)

def remove_outlier(dataframe, col_name):
    
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df2)

num_cols = [col for col in num_cols if col not in "PassengerId"] # ['Age', 'Fare']

# Function calling:
for col in num_cols:
    new_df = remove_outlier(df2, col)

print(df2.shape[0] - new_df.shape[0]) # 116

#------------------------
# 2. Supressing Method (re-assignment with thresholds) 
# Supressing => It is to delete values other than the limit value and write limit values in their place.
#------------------------

low, up = outlier_thresholds(df2, "Fare")

# df2[((df2["Fare"] < low) | (df2["Fare"] > up))]["Fare"]
# alternative code
df2.loc[((df2["Fare"] < low) | (df2["Fare"] > up)), "Fare"]

df2.loc[(df2["Fare"] > up), "Fare"] = up

df2.loc[(df2["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    print(col, check_outlier(df, col))
# Age True
# Fare True

# Function calling for each numerical columns
for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))
#Age False
#Fare False


# Recaping all
###################
# df = load()
# outlier_thresholds(df, "Age")
# check_outlier(df, "Age")
# grab_outliers(df, "Age", index=True)

# remove_outlier(df, "Age").shape
# replace_with_thresholds(df, "Age")
# check_outlier(df, "Age")

#############################################
# Multivariate Outlier Analysis: Local Outlier Factor


# 17, 3 => Variables that are not contradictory on their own but may create contradiction together
# We can call it Multivariate Outliers. (For example, being 17 years old and getting married 3 times)

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()

print(df.head())
print(df.shape) # (53940, 7)

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")

print(df[((df["carat"] < low) | (df["carat"] > up))].shape)
#(1889, 7)

low, up = outlier_thresholds(df, "depth")

print(df[((df["depth"] < low) | (df["depth"] > up))].shape)
# (2545, 7)

clf = LocalOutlierFactor(n_neighbors=20) # class method that produces LOF values 
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
print(df_scores[0:5])
# df_scores = -df_scores => I turned negatives into positives
print(np.sort(df_scores)[0:10])
# Those closer to -1 are good, those closer to -10 are worse outliers

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-') # You can calculate the threshold value from the graph here using the elbow method.
# It can be specified via elbow method and It looks like -5 here.
plt.show()

th = np.sort(df_scores)[3] #  -4.984151747711709 <= Based on the graph, I got the value 3

print(df[df_scores < th])

print(df[df_scores < th].shape)
# (3, 7) => previously thousands of outliers on an individual variable basis were seen,
# but here they decreased to 3 in the multivariate analysis.

print(df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)

print(df[df_scores < th].index)

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index) # outliers deleted

# IMPORTANT NOTE:
# It is okay if I supress when there are a small number of outliers, but if I use supression when there are hundreds of outliers, I will forcibly corrupt 
# the data by creating a lot of duplicate data. So I will have created Noise by this way. In this case, it would be most logical to set my outlier threshold  # values as 0.05 & 0.95 or 0.01 & 0.99 instead of 0.25 & 0.75. Because the number of observations to be trimmed and supressed will be small, which will not    # cause a problem. If I am using tree methods, I should choose not to touch outliers. Because their effects are already low. If I am using linear methods,    # outlier cleaning shoud be done.

#############################################
# 2. MISSING VALUES 

#-------------------------------------
# Catching Missing Values
#-------------------------------------
# df = load()
# df.head()

# Questioning whether there are missing observations or not
# df.isnull().values => returns as a true/false matrix
df.isnull().values.any()

# number of missing values in variables
df.isnull().sum()

# exact number of values in variables
df.notnull().sum()

# total number of missing values in the data set
df.isnull().sum().sum()

# Observation units with at least one missing value
df[df.isnull().any(axis=1)]

# complete observation units
df[df.notnull().all(axis=1)] #[183 rows x 12 columns]

# sorting the missing values in descending order
df.isnull().sum().sort_values(ascending=False)

# I considered the missing values proportionally according to the entire matrix.
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
# ['Age', 'Cabin', 'Embarked']

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

# Function calling:
missing_values_table(df)
#          n_miss  ratio
#Cabin        687  77.10
#Age          177  19.87
#Embarked       2   0.22

missing_values_table(df, True)

#############################################
# Methods to Solve the Missing Value Problem

#-------------------------------------
# Method 1: Quick Delete
#-------------------------------------
df.dropna().shape

#-------------------------------------
# Method 2: Filling with Simple Assignment Methods
#-------------------------------------
# imputing => It can be mean, median or any desired value.
df["Age"].fillna(df["Age"].mean()).isnull().sum()
# df["Age"].fillna(df["Age"].median()).isnull().sum()
# df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0) # axis => 0:row , 1: column 
# df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)
# There are still missing values in some categorical variables. I can use mode() for this.

# filling missing values in a categorical variable column:
# df["Embarked"].mode()[0] => 'S'
# df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
# df["Embarked"].fillna("missing") # I filled it with the any desired expression.

# Filling in missing categorical variables in the entire matrix:
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# len(x.unique()) <= 10 : If the number of unique values is less than 10, I can consider them categorical. Otherwise, I could accept it as a cardinal.

#-------------------------------------
# Assigning Values in Categorical Variable Breakdown

df.groupby("Sex")["Age"].mean()
df["Age"].mean()

# df.groupby("Sex")["Age"].transform("mean") => I filled in the average ages grouped by gender according to the relevant grouping.
# fillna() => It has the feature to capture the breakdown coming from groupby
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#-------------------------------------
# Method 3: Filling with Predictive Assignment
#-------------------------------------
# I will consider the "missing value" variable as the dependent variable and the others as independent variables and perform a modelling.
# I will complete the missing values with the predictions made according to the modeling process.
# KNN => Since it is a distance-based algorithm, I need to standardize.
# df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# I used get_dummies() in the one-hot encoding event here for standardization (GOAL: to express categorical items numerically).
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)  # drop_first=True => For categorical variables with two classes, discard the first and keep the second.
# dff.head()

# Standardization of variables
scaler = MinMaxScaler() # Standardization Class Object
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn application on df
from sklearn.impute import KNNImputer
# KNNImputer => It looks at the nearest valued neighbors, takes the average of the values in the neighbors and assigns them to the missing place.
# It works according to the distance principle.
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# undo the previous standardization and see the actual values: inverse_transform()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[ df["Age"].isnull(), ["Age", "age_imputed_knn"] ]
df.loc[ df["Age"].isnull() ]


###################
# Recapping

# df = load()

# missing table
# missing_values_table(df)

# Calculating numeric variables directly with median
# df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()

# Filling categorical variables with mode
# df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# Filling numeric variables in categorical variable breakdown
# df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()


#############################################
# 3. ADVANCED ANALYSIS

#-------------------------------------
# Examining Missing Data Structure

msno.bar(df) # bar() => Returns non-NaN integers of variables as bar-plot
plt.show()

msno.matrix(df) # matrix() => To obtain information about whether variable missing values occur together
plt.show()

msno.heatmap(df) # heatmap() => nullity correlation values served in here
plt.show()
# At values close to +1; Missing value is thought to occur simultaneously in two dependent variables.
# At values close to -1; It is thought that the missings manifest themselves as being present in one dependent variable but not present in the other.

#-------------------------------------
# Examining the Relationship of Missing Values with the Dependent Variable

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns 

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)

#############################################
# 4. ENCODING (Label Encoding, One-Hot Encoding, Rare Encoding)

#-------------------------------------
# A. Label Encoding & Binary Encoding
#---------------------------------------

# df = load()
# df.head()
df["Sex"].head()

le = LabelEncoder() 
le.fit_transform(df["Sex"])[0:5] # array([1, 0, 0, 0, 1]) => In alphabatical order, the first one (female) is 0.
le.inverse_transform([0, 1]) # array(['female', 'male'], dtype=object) => It can be used to remember conversion information.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    
    return dataframe

# If I have hundreds of variables, I use the below code line to select binary columns (categorical variables):
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

#---------------------------------------
# df = load_application_train()
# df.shape

# binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
#                and df[col].nunique() == 2]

# df[binary_cols].head() # After the conversion, we want to see 0s and 1s, but
# If you observe 2, these represent missing values. Watch out this!

# for col in binary_cols:
#     label_encoder(df, col)


# df["Embarked"].value_counts()
# df["Embarked"].nunique()
# len(df["Embarked"].unique())
# nunique() vs unique() : It is important to be aware of the distinction!! unique() => Includes NaNs.

#---------------------------------------
# B.One-Hot Encoding
#---------------------------------------

# df = load()
# df.head()
df["Embarked"].value_counts()

# pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head() #dummy_na=True =>  It also creates a class for missing values.

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head() # Here, I can do both binary encode and one-hot encode.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# cat_cols, num_cols, cat_but_car = grab_col_names(df) => alternative for below code line
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()
# df.head()

#---------------------------------------
# C. Rare Encoding
#---------------------------------------
# 1. Analyzing the abundance of categorical variables.
# 2. Analyzing the relationship between rare categories and the dependent variable.
# 3. Rare encoder function code.

#---------------------------------------
# 1. Analyzing the abundance of categorical variables

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# cat_summary => shows the classes and proportions of categorical variables
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# I observed unnecessary classes with low frequency under categorical variables, I should remove them
# or I must continue with rare encoding.

#---------------------------------------
# 2. Analyzing the relationship between rare categories and the dependent variable

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# target => dependent variable, cat_cols => categoric variable
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts())) # information on how many classes there are
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)
# If I use the following as an example, I can collect the items below 1% as Rare.
# DEF_60_CNT_SOCIAL_CIRCLE : 9
#                            COUNT     RATIO  TARGET_MEAN
# DEF_60_CNT_SOCIAL_CIRCLE
# 0.0                       280721  0.912881     0.078348
# 1.0                        21841  0.071025     0.105169
# 2.0                         3170  0.010309     0.121451
# Alttakiler RARE olabilir.
# 3.0                          598  0.001945     0.158863
# 4.0                          135  0.000439     0.111111
# 5.0                           20  0.000065     0.150000
# 6.0                            3  0.000010     0.000000
# 7.0                            1  0.000003     0.000000
# 24.0                           1  0.000003     0.000000

#---------------------------------------
# 3. Rare encoder'function

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# It will bring together classes of categorical variables that fall below the 0.01 ratio.
new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#############################################
# 5.FEATURE SCALING 

#-----------------------
# StandardScaler: Classic standardization. Calculate the mean, divide by the standard deviation. z = (x - u) / s
# There is influence of missing and outlier values.

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#-----------------------
# RobustScaler: Calculate the median and divide by the IQR.
# More reliable (unaffected by outliers) than StandardScaler. But for some reason it is not widely used in the market.

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#-----------------------
# MinMaxScaler: Variable conversion between two given values is widely used. It is very used for a special range

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

# I used it to show quarterly values of numerical variables and create graphs
# to see the big picture
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

#-----------------------
# Numeric to Categorical:Converting Numeric Variables to Categorical Variables
# Binning


df["Age_qcut"] = pd.qcut(df['Age'], 5)

#############################################
# 6.FEATURE EXTRACTION 
# Binary Features: Flag, Bool, True-False => Deriving new variables from existing variables

# df = load()
# df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
#Cabin   NEW_CABIN_BOOL
#NaN             0
#C85             1
#NaN             0

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
#                 Survived
# NEW_CABIN_BOOL
# 0                  0.300
# 1                  0.667


from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 9.4597, p-value = 0.0000
# H0 is rejected, so it can be said that there is a significant difference between them.

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})
#               Survived
# NEW_IS_ALONE
# NO               0.506
# YES              0.304


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = -6.0704, p-value = 0.0000
# H0 is rejected, so it can be said that there is a significant difference between them.

#############################################
# Deriving Features from Texts

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Capturing Special Structures

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})
#                 mean count
# NEW_NAME_DR
# 0              0.383   881
# 1              0.500    10

###################
# Deriving a Variable with REGEX

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
#         Survived   Age
#               mean count   mean
# NEW_TITLE
# Capt         0.000     1 70.000
# Col          0.500     2 58.000
# Countess     1.000     1 33.000
# Don          0.000     1 40.000
# Dr           0.429     6 42.000
# Jonkheer     0.000     1 38.000
# Lady         1.000     1 48.000
# Major        0.500     2 48.500
# Master       0.575    36  4.574
# Miss         0.698   146 21.774
# Mlle         1.000     2 24.000
# Mme          1.000     1 24.000
# Mr           0.157   398 32.368
# Mrs          0.792   108 35.898
# Ms           1.000     1 28.000
# Rev          0.000     6 43.167
# Sir          1.000     1 49.000

#############################################
# Deriving Date Variables

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff => (month difference between two dates): year difference + month difference
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

#############################################
# 7. FEATURE INTERACTIONS  : It means that variables interact with each other.

# df = load()
# df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 
# 1 => the person himself

df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.groupby("NEW_SEX_CAT")["Survived"].mean()
