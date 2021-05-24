import itertools
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from SeqMixModel_All_ContantNoGRU_Sample import AggSeqModel_pair
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import convert_unique_idx, AllFeaturePair_Constant_id_Sample

warnings.filterwarnings('ignore')


def movie_preprocessing(movie):
    movie_col = list(movie.columns)
    movie_tag = [doc.split('|') for doc in movie['genres']]
    tag_table = {token: idx for idx, token in enumerate(set(itertools.chain.from_iterable(movie_tag)))}
    movie_tag = pd.DataFrame(movie_tag)
    tag_table = pd.DataFrame(tag_table.items())
    tag_table.columns = ['Tag', 'Index']

    # use one-hot encoding for movie genres (here called tag)
    tag_dummy = np.zeros([len(movie), len(tag_table)])

    for i in tqdm(range(len(movie))):
        for j in range(len(tag_table)):
            if tag_table['Tag'][j] in list(movie_tag.iloc[i, :]):
                tag_dummy[i, j] = 1

    # combine the tag_dummy one-hot encoding table to original movie files
    movie = pd.concat([movie, pd.DataFrame(tag_dummy)], 1)
    movie_col.extend(['tag' + str(i) for i in range(len(tag_table))])
    movie.columns = movie_col
    movie = movie.drop('genres', 1)
    return movie


users_df = pd.read_csv('../dataset/ml-1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'],
                       engine="python")

users_df['user_id'] = users_df['user_id'].astype(np.int)
users_df['age'] = users_df['age'].astype(np.int)
users_df['occupation'] = users_df['occupation'].astype(np.int)
users_df['zip'] = users_df['zip'].astype(str)

movies_df = pd.read_csv('../dataset/ml-1m/movies.dat', sep='::', names=['item_id', 'title', 'genres'], engine="python")
movies_df['item_id'] = movies_df['item_id'].astype(np.int)
movies_df = movie_preprocessing(movies_df)

ratings_df = pd.read_csv('../dataset/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'],
                         engine="python")
ratings_df['user_id'] = ratings_df['user_id'].astype(np.int)
ratings_df['item_id'] = ratings_df['item_id'].astype(np.int)

data_df = pd.merge(ratings_df, users_df, how="inner", on="user_id")
data_df = pd.merge(data_df, movies_df, how="inner", on="item_id")

data_df = data_df[data_df["rating"] != 3]
data_df, user_mapping = convert_unique_idx(data_df, "user_id")
data_df, item_mapping = convert_unique_idx(data_df, "item_id")
data_df["label"] = data_df["rating"].apply(lambda x: 1 if x > 3 else 0)

sparse_features = ["item_id", "user_id",
                   "gender", "age", "occupation", "zip"]

for feat in sparse_features:
    lbe = LabelEncoder()
    data_df[feat] = lbe.fit_transform(data_df[feat])

data_df.iloc[:, 0] = data_df["user_id"] + 1
data_df.iloc[:, 1] = data_df["item_id"] + 1

from sklearn.model_selection import train_test_split

train_df, rest_df = train_test_split(data_df, test_size=0.2)
comItems = np.intersect1d(rest_df["item_id"].unique().tolist(), train_df["item_id"].unique().tolist())
rest_df = rest_df[rest_df["item_id"].isin(comItems)]
assert len(np.intersect1d(rest_df["item_id"].unique().tolist(), train_df["item_id"].unique().tolist())) == rest_df[
    "item_id"].nunique()

test_df, valid_df = train_test_split(rest_df, test_size=0.5)

num_user = data_df["user_id"].nunique()
num_item = data_df["item_id"].nunique()
num_gender = data_df["gender"].nunique()
num_age = data_df["age"].nunique()
num_occupation = data_df["occupation"].nunique()
num_zip = data_df["zip"].nunique()
num_genres = 18

user_dict = dict()
for index, pdf in tqdm(train_df.groupby(["user_id"])):
    tem_dict = dict()
    rating_list = pdf["rating"].values.tolist()
    count = Counter(rating_list)
    key_list = list(count.keys())
    value_list = list(count.values())
    for k in key_list:
        tem_dict[k] = pdf[pdf["rating"] == k]["item_id"].values.tolist()
    user_dict[index] = tem_dict

item_dict = dict()
for index, pdf in tqdm(train_df.groupby(["item_id"])):
    tem_dict = dict()
    rating_list = pdf["rating"].values.tolist()
    count = Counter(rating_list)
    key_list = list(count.keys())
    value_list = list(count.values())
    for k in key_list:
        tem_dict[k] = pdf[pdf["rating"] == k]["user_id"].values.tolist()
    item_dict[index] = tem_dict

dataset = AllFeaturePair_Constant_id_Sample(train_df, sequence_len=20, user_dict=user_dict, item_dict=item_dict)
data_loader = DataLoader(dataset, batch_size=1024, num_workers=32)

params = dict(
    input_size=200,
    hidden_size=256,
    num_ratings=5,
    num_layers=3,
    dropout=0.3,
    learning_rate=0.0001,
    epoch=130,
    sp_hidden_size=56,
    data_loader=data_loader,
    num_users=num_user,
    num_items=num_item,
    num_gender=num_gender,
    num_age=num_age,
    num_occupation=num_occupation,
    num_zip=num_zip,
    num_genre=num_genres,
    test_df=test_df,
    valid_df=valid_df,
    user_dict=user_dict,
    item_dict=item_dict,
    seq_len=20,
    save=False,
    name="all_feature_ml1m",
    load=False,
)

model = AggSeqModel_pair(**params)

model.run()

r = model.results

auc_valid = [i[0] for i in r]
llog_valid = [i[1] for i in r]

auc_test = [i[2] for i in r]
llog_test = [i[3] for i in r]

print(f"auc score on valid: {max(auc_valid)}")
print(f"logloss score on valid: {min(llog_valid)}")

print(f"auc score on test: {max(auc_test)}")
print(f"logloss score on test: {min(llog_test)}")
