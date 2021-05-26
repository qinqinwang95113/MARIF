import pandas as pd
import os
import numpy as np
import random
import math
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple
from collections import Counter


class DatasetLoader(object):
    def load(self):
        raise NotImplementedError


class ML1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, "ratings.dat")
        self.genre_path = os.path.join(data_dir, "movies.dat")

    def load(self):
        df = pd.read_csv(
            self.fpath,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"]
        )

        return df

    def load_with_genre(self):
        ratings_df = self.load()
        genre_df = pd.read_csv(
            self.genre_path,
            sep="::",
            header=None,
            names=['item_id', 'movie_title', 'movie_genre']
        )
        genre_df = pd.concat([genre_df, genre_df.movie_genre.str.get_dummies(sep='|')], axis=1)
        # drop title and original genre rep
        genre_df = genre_df.drop(["movie_title", "movie_genre"], axis=1)
        merged_df = ratings_df.merge(genre_df, on="item_id")
        return merged_df

class ML20M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, "ratings.csv")

    def load(self):
        df = pd.read_csv(
            self.fpath,
            sep=",",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"]
        )
        return df


class TriplePair(Dataset):
    def __init__(self, user_dict, item_dict, num_item, user_list, pair):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.user_list = user_list
        self.pair = pair
        self.num_item = num_item

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        uid = self.pair[idx][0]
        iid = self.pair[idx][1]
        jid = np.random.randint(self.num_item)
        while jid in self.user_list[uid]:
            jid = np.random.randint(self.num_item)

        u = self.user_dict[uid][np.random.choice(len(self.user_dict[uid]))]
        i = self.item_dict[iid][np.random.choice(len(self.item_dict[iid]))]
        j = self.item_dict[jid][np.random.choice(len(self.item_dict[jid]))]

        return t.LongTensor(u), t.LongTensor(i), t.LongTensor(j)


class TriplePairLabel(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, 4]]

        u = self.user_dict[uid][np.random.choice(len(self.user_dict[uid]))]
        i = self.item_dict[iid][np.random.choice(len(self.item_dict[iid]))]

        return t.LongTensor(u), t.LongTensor(i), t.LongTensor(label)


class TriplePairLabel_last(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, 4]]

        u = self.user_dict[uid]
        i = self.item_dict[iid]

        return t.LongTensor(u), t.LongTensor(i), t.LongTensor(label)


class TriplePairLabel_rand(Dataset):
    def __init__(self, user_dict, item_dict, raw_df, window_size):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df
        self.window_size = window_size

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, 4]]

        if (len(self.user_dict[uid]) > self.window_size):
            start_index_u = np.random.choice(len(self.user_dict[uid]) - self.window_size)
            u = self.user_dict[uid][start_index_u: start_index_u + self.window_size]
        else:
            u = self.user_dict[uid]

        if (len(self.item_dict[iid]) > self.window_size):
            start_index_i = np.random.choice(len(self.item_dict[iid]) - self.window_size)
            i = self.item_dict[iid][start_index_i: start_index_i + self.window_size]
        else:
            i = self.item_dict[iid]

        return t.LongTensor(u), t.LongTensor(i), t.LongTensor(label)


class SimpleMF(Dataset):
    def __init__(self, train_df):
        self.train_df = train_df

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        uid = [self.train_df.iloc[idx, 0]]
        iid = [self.train_df.iloc[idx, 1]]
        label = [self.train_df.iloc[idx, 4]]

        return t.LongTensor(uid), t.LongTensor(iid), t.LongTensor(label)


class BaseSession(Dataset):
    def __init__(self, train_df, user_dict, item_dict):
        self.uid_to_index = dict(zip(list(user_dict.keys()), list(range(len(user_dict.keys())))))
        self.iid_to_index = dict(zip(list(item_dict.keys()), list(range(len(item_dict.keys())))))

        self.train_df = train_df
        useq = [u for u in user_dict.values()]
        iseq = [i for i in item_dict.values()]
        self.pad_user_dict = pad_sequences(useq)
        self.pad_item_dict = pad_sequences(iseq)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        uid = self.train_df.iloc[idx, 0]
        iid = self.train_df.iloc[idx, 1]
        uid = self.uid_to_index[uid]
        iid = self.iid_to_index[iid]

        label = [self.train_df.iloc[idx, 4]]
        user_seq = self.pad_user_dict[uid]
        item_seq = self.pad_item_dict[iid]

        return t.LongTensor(user_seq), t.LongTensor(item_seq), t.LongTensor(label)


class BaseSessionSample(Dataset):
    def __init__(self, train_df, user_dict, item_dict):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.train_df = train_df

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        uid = self.train_df.iloc[idx, 0]
        iid = self.train_df.iloc[idx, 1]

        label = [self.train_df.iloc[idx, 4]]

        user_idx = np.random.choice(len(self.user_dict[uid]))
        item_idx = np.random.choice(len(self.item_dict[iid]))

        user_seq = self.user_dict[uid][user_idx]
        item_seq = self.item_dict[iid][item_idx]

        return t.LongTensor(user_seq), t.LongTensor(item_seq), t.LongTensor(label)


class HalfMF(Dataset):
    def __init__(self, item_dict, train_df):
        self.item_dict = item_dict
        self.train_df = train_df

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        uid = [self.train_df.iloc[idx, 0]]
        iid_tem = self.train_df.iloc[idx, 1]
        label = [self.train_df.iloc[idx, 4]]

        iid = self.item_dict[iid_tem][np.random.choice(len(self.item_dict[iid_tem]))]

        return t.LongTensor(uid), t.LongTensor(iid), t.LongTensor(label)


class MixPair(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, 4]]

        u = self.user_dict[uid][np.random.choice(len(self.user_dict[uid]))]
        i = self.item_dict[iid][np.random.choice(len(self.item_dict[iid]))]

        return t.LongTensor(u), t.LongTensor(i), t.LongTensor([uid]), t.LongTensor([iid]), t.LongTensor(label)


# sequence mix pair
class SeqMixPair(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        user_idx = np.random.choice(len(self.user_dict[uid][0]))
        item_idx = np.random.choice(len(self.item_dict[iid][0]))

        item_seq_ids = self.user_dict[uid][1][user_idx]
        u_seq_ratings = self.user_dict[uid][0][user_idx]

        user_seq_ids = self.item_dict[iid][1][item_idx]
        i_seq_ratings = self.item_dict[iid][0][item_idx]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(label)


# sequence all feature's pair *****************
class AllFeaturePair(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        gender = [self.raw_df.iloc[idx, 4]]
        age = [self.raw_df.iloc[idx, 5]]
        occupation = [self.raw_df.iloc[idx, 6]]
        zipCode = [self.raw_df.iloc[idx, 7]]
        genres = self.raw_df.iloc[idx, 9:-1]

        user_idx = np.random.choice(len(self.user_dict[uid][0]))
        item_idx = np.random.choice(len(self.item_dict[iid][0]))

        item_seq_ids = self.user_dict[uid][1][user_idx]
        u_seq_ratings = self.user_dict[uid][0][user_idx]

        user_seq_ids = self.item_dict[iid][1][item_idx]
        i_seq_ratings = self.item_dict[iid][0][item_idx]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(gender), t.LongTensor(age), t.LongTensor(occupation), t.LongTensor(
            zipCode), t.LongTensor(genres), t.LongTensor(label)


class AllFeaturePair_Constant_id(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        gender = [self.raw_df.iloc[idx, 4]]
        age = [self.raw_df.iloc[idx, 5]]
        occupation = [self.raw_df.iloc[idx, 6]]
        zipCode = [self.raw_df.iloc[idx, 7]]
        genres = self.raw_df.iloc[idx, 9:-1]

        item_seq_ids = self.user_dict[uid][1]
        u_seq_ratings = self.user_dict[uid][0]

        user_seq_ids = self.item_dict[iid][1]
        i_seq_ratings = self.item_dict[iid][0]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(gender), t.LongTensor(age), t.LongTensor(occupation), t.LongTensor(
            zipCode), t.LongTensor(genres), t.LongTensor(label)



class AllFeaturePair_Constant_id_Sample(Dataset):
    def __init__(self, raw_df, sequence_len, user_dict, item_dict):
        self.raw_df = raw_df
        self.seq_len = sequence_len
        self.user_dict = user_dict
        self.item_dict = item_dict

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        gender = [self.raw_df.iloc[idx, 4]]
        age = [self.raw_df.iloc[idx, 5]]
        occupation = [self.raw_df.iloc[idx, 6]]
        zipCode = [self.raw_df.iloc[idx, 7]]
        genres = self.raw_df.iloc[idx, 9:-1]

        u_seq_ratings, item_seq_ids = self.up_down_sampling(self.user_dict[uid], "item_id")
        i_seq_ratings, user_seq_ids = self.up_down_sampling(self.item_dict[iid], "user_id")

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(gender), t.LongTensor(age), t.LongTensor(occupation), t.LongTensor(
            zipCode), t.LongTensor(genres), t.LongTensor(label)

    def up_down_sampling(self, uidict, unaspect):
        tem_list_rating = []
        tem_list_id = []
        key_list = list(uidict.keys())
        value_list = [len(uidict[k]) for k in uidict]
        sum_val = sum(value_list)
        for i in range(len(key_list) - 1):
            rp_times = int(self.seq_len * value_list[i] / sum_val)
            tem_list_rating.extend(np.repeat(key_list[i], rp_times).tolist())
            tem_list_id.extend([np.random.choice(uidict[key_list[i]]) for _ in range(rp_times)])
        left_num = self.seq_len - len(tem_list_rating)
        tem_list_rating.extend(np.repeat(key_list[-1], left_num).tolist())
        tem_list_id.extend([np.random.choice(uidict[key_list[-1]]) for _ in range(left_num)])

        ri_list = list(zip(tem_list_rating, tem_list_id))
        np.random.shuffle(ri_list)
        list_rating = [i[0] for i in ri_list]
        list_id = [i[1] for i in ri_list]
        return list_rating, list_id


class AllFeaturePair_Constant_lastFM(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        item_seq_ids = self.user_dict[uid][1]
        u_seq_ratings = self.user_dict[uid][0]

        user_seq_ids = self.item_dict[iid][1]
        i_seq_ratings = self.item_dict[iid][0]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(label)


class AllFeaturePair_Constant(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        gender = [self.raw_df.iloc[idx, 4]]
        age = [self.raw_df.iloc[idx, 5]]
        occupation = [self.raw_df.iloc[idx, 6]]
        zipCode = [self.raw_df.iloc[idx, 7]]
        genres = self.raw_df.iloc[idx, 9:-1]

        u_seq_ratings = self.user_dict[uid]

        i_seq_ratings = self.item_dict[iid]

        return t.LongTensor(u_seq_ratings), t.LongTensor(i_seq_ratings), t.LongTensor(gender), t.LongTensor(
            age), t.LongTensor(occupation), t.LongTensor(zipCode), t.LongTensor(genres), t.LongTensor(label)


class AllFeaturePair_Amazon(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        style = [self.raw_df.iloc[idx, 3]]
        price = [self.raw_df.iloc[idx, 6]]

        user_idx = np.random.choice(len(self.user_dict[uid][0]))
        item_idx = np.random.choice(len(self.item_dict[iid][0]))

        item_seq_ids = self.user_dict[uid][1][user_idx]
        u_seq_ratings = self.user_dict[uid][0][user_idx]

        user_seq_ids = self.item_dict[iid][1][item_idx]
        i_seq_ratings = self.item_dict[iid][0][item_idx]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(style), t.LongTensor(price), t.LongTensor(label)


class AllFeaturePair_Amazon_Constant_id(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        style = [self.raw_df.iloc[idx, 3]]
        price = [self.raw_df.iloc[idx, 6]]

        item_seq_ids = self.user_dict[uid][1]
        u_seq_ratings = self.user_dict[uid][0]

        user_seq_ids = self.item_dict[iid][1]
        i_seq_ratings = self.item_dict[iid][0]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(style), t.LongTensor(price), t.LongTensor(label)


class AllFeaturePair_ML20M(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        genres = self.raw_df.iloc[idx, 5:-1]

        user_idx = np.random.choice(len(self.user_dict[uid][0]))
        item_idx = np.random.choice(len(self.item_dict[iid][0]))

        item_seq_ids = self.user_dict[uid][1][user_idx]
        u_seq_ratings = self.user_dict[uid][0][user_idx]

        user_seq_ids = self.item_dict[iid][1][item_idx]
        i_seq_ratings = self.item_dict[iid][0][item_idx]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(genres), t.LongTensor(label)


class AllFeaturePair_ConstantId_ML20M(Dataset):
    def __init__(self, user_dict, item_dict, raw_df):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.raw_df = raw_df

    def __len__(self):
        return len(self.raw_df)

    def __getitem__(self, idx):
        #         idx = np.random.randint(len(self.pair))
        # df: user id, item id, rating, timestamp, label
        uid = self.raw_df.iloc[idx, 0]
        iid = self.raw_df.iloc[idx, 1]
        label = [self.raw_df.iloc[idx, -1]]

        genres = self.raw_df.iloc[idx, 5:-1]

        item_seq_ids = self.user_dict[uid][1]
        u_seq_ratings = self.user_dict[uid][0]

        user_seq_ids = self.item_dict[iid][1]
        i_seq_ratings = self.item_dict[iid][0]

        return t.LongTensor(u_seq_ratings), t.LongTensor(item_seq_ids), t.LongTensor(i_seq_ratings), t.LongTensor(
            user_seq_ids), t.LongTensor(genres), t.LongTensor(label)


# emd generator data
class EmdGen(Dataset):
    def __init__(self, emd, seq_dict):
        self.emd = emd
        self.seq_emd, self.map = seq_dict["emd"], seq_dict["map_dict"]
        self.ids = np.array(list(self.map.keys()))

    def __len__(self):
        return len(self.map)

    def __getitem__(self, index):
        out_emd = self.emd[index]

        key = self.ids[index]
        seq_emd_index = self.map[key]
        out_seq_emd = self.seq_emd[seq_emd_index]
        return out_emd, out_seq_emd


class SiameseEmdDataset(Dataset):
    def __init__(self, emd, seqEmd_dict):
        self.emd = emd
        self.seqEmd_dict = seqEmd_dict
        self.numEmd_samples = emd.shape[0]
        # sample step
        self.pair = []
        print("create pos/neg pair list 1:1")
        for i in tqdm(self.seqEmd_dict):
            id_emd = self.emd[i]
            for seq_emd in self.seqEmd_dict[i]:
                # positive
                self.pair.append([1, id_emd, seq_emd])
                # negative
                n_id = np.random.choice(range(self.numEmd_samples))
                while n_id == i:
                    n_id = np.random.choice(range(self.numEmd_samples))
                self.pair.append([0, self.emd[n_id], seq_emd])

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        label = [self.pair[idx][0]]  # type:int need to be flatten
        id_emd = self.pair[idx][1]  # np.array(float32)
        seq_emd = self.pair[idx][2]  # np.array(float32)

        return t.LongTensor(label), t.LongTensor(id_emd), t.LongTensor(seq_emd)


class PentaPair(Dataset):
    def __init__(self, user_dict, item_dict, num_item, user_list, pair, genre_mat):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.user_list = user_list
        self.pair = pair
        self.num_item = num_item
        self.genre_mat = genre_mat

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        uid = self.pair[idx][0]
        iid = self.pair[idx][1]
        jid = np.random.randint(self.num_item)
        while jid in self.user_list[uid]:
            jid = np.random.randint(self.num_item)

        u = self.user_dict[uid][np.random.choice(len(self.user_dict[uid]))]
        i = self.item_dict[iid][np.random.choice(len(self.item_dict[iid]))]
        j = self.item_dict[jid][np.random.choice(len(self.item_dict[jid]))]

        genre_i = self.genre_mat[iid]
        genre_j = self.genre_mat[jid]

        return t.LongTensor(u), t.LongTensor(i), t.LongTensor(j), t.LongTensor(genre_i), t.LongTensor(genre_j)


def convert_unique_idx(df, column_name):
    column_dic = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dic.get)
    df[column_name] = df[column_name].astype("int")
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dic) - 1

    return df, column_dic


##########################################################################################

def create_user_list(df, user_size):
    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user_id].append((row.timestamp, row.item_id))
    return user_list


def create_user_list_explicit(df, user_size):
    user_list = [list() for u in range(user_size)]
    for row in df[df["label"] == 1].itertuples():
        user_list[row.user_id].append(row.item_id)
    return user_list


def explicit_train_test_split(df, user_size, test_size=0.2, time_order=False):
    test_id = np.random.choice(len(df), size=int(len(df) * test_size))
    train_id = list(set(range(len(df))) - set(test_id))
    test_df = df.loc[test_id].reset_index(drop=True)
    train_df = df.loc[train_id].reset_index(drop=True)

    test_user_list = explicit_create_user_list(test_df, user_size)
    train_user_list = explicit_create_user_list(train_df, user_size)

    return train_user_list, test_user_list


##########################################################################################
def train_test_split(df, user_size, test_size=0.2, time_order=False):
    if not time_order:
        test_id = np.random.choice(len(df), size=int(len(df) * test_size))
        train_id = list(set(range(len(df))) - set(test_id))
        test_df = df.loc[test_id].reset_index(drop=True)
        train_df = df.loc[train_id].reset_index(drop=True)

        test_user_list = create_user_list(test_df, user_size)
        train_user_list = create_user_list(train_df, user_size)

    else:
        total_user_list = create_user_list(df, user_size)
        train_user_list = [list() for u in range(user_size)]
        test_user_list = [list() for u in range(user_size)]

        for user, item_list in enumerate(total_user_list):
            # choose the latest items
            item_list = sorted(item_list, key=lambda x: x[0])
            # split items
            test_item = item_list[math.ceil(len(item_list) * (1 - test_size)):]
            train_item = item_list[:math.ceil(len(item_list) * (1 - test_size))]
            # register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item

        # remove the timestamp
        test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
        train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
        return train_user_list, test_user_list


def explicit_create_tri_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend()


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def create_dict(df, aspect, seq_len, padding=0):
    tem_dict = dict()
    for index, pdf in tqdm(df.groupby([aspect])):
        pdf = pdf.sort_values("timestamp")
        seq_list = []
        num_seq = int(np.floor(len(pdf) / seq_len))
        seq_list = [pdf["rating"].values[i * seq_len: (i + 1) * seq_len].tolist() for i in range(num_seq)]
        # permute the sequence
        #         seq_list = np.random.permutation(seq_list).tolist()
        ######
        last_seq = list(pdf["rating"].values[num_seq * seq_len:])
        if (len(pdf) % seq_len != 0):
            padding_len = ((num_seq + 1) * seq_len) - len(pdf)
            last_seq.extend([0 for i in range(padding_len)])
            seq_list.append(last_seq)
        tem_dict[index] = seq_list
    return tem_dict


def create_mix_dict(df, aspect, unaspect, seq_len, padding=0):
    tem_dict = dict()
    for index, pdf in tqdm(df.groupby([aspect])):
        pdf = pdf.sort_values("timestamp")
        seq_list = []
        num_seq = int(np.floor(len(pdf) / seq_len))
        seq_list = [pdf["rating"].values[i * seq_len: (i + 1) * seq_len].tolist() for i in range(num_seq)]
        id_list = [pdf[unaspect].values[i * seq_len: (i + 1) * seq_len].tolist() for i in range(num_seq)]
        last_seq = list(pdf["rating"].values[num_seq * seq_len:])
        last_id = list(pdf[unaspect].values[num_seq * seq_len:])
        if (len(pdf) % seq_len != 0):
            padding_len = ((num_seq + 1) * seq_len) - len(pdf)
            last_seq.extend([0 for i in range(padding_len)])
            last_id.extend([0 for i in range(padding_len)])
            seq_list.append(last_seq)
            id_list.append(last_id)
        tem_dict[index] = (seq_list, id_list)
    return tem_dict


def create_constant_mix_dict(df, aspect, unaspect, seq_len, paddding=0):
    tem_dict = dict()
    for index, pdf in tqdm(df.groupby([aspect])):
        tem_list_rating = []
        tem_list_id = []
        rating_list = pdf["rating"].values.tolist()
        count = Counter(rating_list)
        key_list = list(count.keys())
        value_list = list(count.values())
        sum_val = sum(value_list)
        for i in range(len(key_list) - 1):
            rp_times = int(seq_len * value_list[i] / sum_val)
            tem_list_rating.extend(np.repeat(key_list[i], rp_times).tolist())
            tem_list_id.extend(
                [np.random.choice(pdf[pdf["rating"] == key_list[i]][unaspect].values) for _ in range(rp_times)])
        left_num = seq_len - len(tem_list_rating)
        tem_list_rating.extend(np.repeat(key_list[-1], left_num).tolist())
        tem_list_id.extend(
            [np.random.choice(pdf[pdf["rating"] == key_list[-1]][unaspect].values) for _ in range(left_num)])

        ri_list = list(zip(tem_list_rating, tem_list_id))
        np.random.shuffle(ri_list)
        list_rating = [i[0] for i in ri_list]
        list_id = [i[1] for i in ri_list]
        tem_dict[index] = (list_rating, list_id)
    return tem_dict


def create_fmax_pool_dict(df, aspect, seq_len, padding):
    tem_dict = {}
    for index, pdf in tqdm(df.groupby([aspect])):
        pdf = pdf.sort_values("timestamp")
        rating_list = pdf["rating"].values.tolist()
        if (len(rating_list) < seq_len):
            pool_list = [padding for i in range(seq_len - len(rating_list))]
            pool_list.extend(rating_list)
        else:
            window_size = len(rating_list) - seq_len + 1
            pool_list = flex_mean_pooling(rating_list, window_size)
        tem_dict[index] = pool_list
    return tem_dict


#################################### Sparse Feature ######################################################
def flex_maxmin_pooling(sequence, window_size):
    start_index = 0
    end_idnex = window_size
    pooling_list = []
    for i in range(len(sequence) - window_size + 1):
        max_val = max(sequence[start_index: end_idnex])
        min_val = min(sequence[start_index: end_idnex])
        val = np.random.choice([max_val, min_val])
        pooling_list.append(val)
        start_index += 1
        end_idnex += 1
    return pooling_list


def flex_mean_pooling(sequence, window_size):
    start_index = 0
    end_idnex = window_size
    pooling_list = []
    for i in range(len(sequence) - window_size + 1):
        val = round(np.mean(sequence[start_index: end_idnex]))
        pooling_list.append(val)
        start_index += 1
        end_idnex += 1
    return pooling_list


class SparseFeature(namedtuple("SparseFeature",
                               ["name", "total_num", "emd_size"])):
    __slots__ = ()

    def __new__(cls, name, total_num, emd_size=32):
        return super(SparseFeaturem, cls).__new__(cls, name, total_num, emd_size)





