# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

train = pd.read_csv(r"C:\Users\gaume\Documents\kaggle\Titanic\data\train.csv")

test = pd.read_csv(r"C:\Users\gaume\Documents\kaggle\Titanic\data\test.csv")

gender_submission = pd.read_csv(r"C:\Users\gaume\Documents\kaggle\Titanic\data\gender_submission.csv")

# +
# gender_submission

# +
# train

# +
# test
# -

# lower case column names
train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()

# +
# train
# -

train[["fist_name", "rest_name"]] = train.name.str.split(",", expand = True)

train[["title", "last_name", "other"]]=train["rest_name"].str.split(".", expand = True)

# +
# train[["title", "last_name", "other"]]

# +
# train[["fist_name", "title", "last_name", "other"]]
# -

# look at the servival rate for each gender
train.groupby("sex")["survived"].describe()

# strip spaces from title
train["title"] = train["title"].str.strip()

train["title"].value_counts()

train["title"] = train["title"].replace(["Dr", "Rev", "Major", "Mlle", "Col", "Capt", "Mme", "Sir", "Ms", "Jonkheer", "the Countess", "Don", "Lady"],"other")#.value_counts()

# look at the servival rate for each title
train.groupby("title")["survived"].describe()

train["age_group"] = pd.cut(x=train["age"], bins = [0,18,30,40,50,65,120])

train["age_group"].value_counts()

train["age_group_number"] = train["age_group"].apply(lambda x: str(x)).replace(["(0, 18]", "(18, 30]", "(30, 40]", "(40, 50]", "(50, 65]", "(65, 120]"],[9, 24, 35, 45, 58, 73])# value_counts()

train["age_group_str"] = train["age_group"].apply(lambda x: str(x)).replace(["(0, 18]", "(18, 30]", "(30, 40]", "(40, 50]", "(50, 65]", "(65, 120]"],["0-18", "18-30", "30-40", "40-50", "50-65", "65-80"])# value_counts()

# look at the servival rate for each age groupb
train.groupby("age_group_str")["survived"].describe()

train["sibsp"].value_counts()

train["cabin"].describe()

train["cabin_first_letter"] = train["cabin"].apply(lambda x: x if pd.isna(x) else x[:1])

train["cabin_first_letter"].value_counts()

# look at the servival rate for each cabin
train.groupby("cabin_first_letter")["survived"].describe()

train.columns

train.isna().sum()

train.groupby("title")["age"].mean()

train["age"].fillna(value = train["age"].mean(), inplace = True)

train["age"].mean()

train.isna().sum()

train["cabin_first_letter"].value_counts()

len(train)


