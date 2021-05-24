import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch as t
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
import pathlib
from collections import Counter

# from deepTrans.utils import EmdGen
# from utils import EmdGen
# t.multiprocessing.set_sharing_strategy('file_system')





t.cuda.set_device(1)
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# set cuda device

print(f"=>using {device}")
# device = t.device("cpu")

# This is gauc score which used in DIN model
def calc_auc(preds, labels):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    raw_arr = []
    for p ,t in zip(preds, labels):
        raw_arr.append([p, t])
    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc



class Feature_Extractor(nn.Module):
    def __init__(self, input_size, num_ratings, hidden_size, num_layers, num_ui):
        super(Feature_Extractor, self).__init__()
        # embedding for id(with bias or not??)
        self.id_emd = nn.Parameter(t.empty(num_ui + 1, input_size))
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ratings_emd = nn.Parameter(t.empty(num_ratings + 1, input_size))
        nn.init.xavier_normal_(self.ratings_emd)
        nn.init.xavier_normal_(self.id_emd)
#         self.ratings_emd = nn.Embedding(num_ratings + 1, input_size)
#         self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True, num_layers = num_layers, bidirectional = False)
        # the input is the concatenation of id emd and rating sequence emd
#         self.gru = nn.GRU(input_size = input_size * 2, hidden_size = hidden_size, batch_first = True, num_layers = num_layers, bidirectional = False)
        self.projection_layer = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.ReLU(),
        )
    
        
        
    
    def forward(self, x, ids):
        x = self.ratings_emd[x, :]
        x_ids = self.id_emd[ids, :]
        x_all = t.cat((x, x_ids), dim = 2)
#         h0 = t.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(device)
#         c0 = t.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(device)
#         out, _ = self.gru(x_all, h0)
        out = self.projection_layer(x_all)
        out = out.mean(axis = 1)
        return out
    
    
    
class SparseFeature(nn.Module):
    def __init__(self, total_num, emd_size):
        super(SparseFeature, self).__init__()
        self.emd = nn.Parameter(t.empty(total_num, emd_size))
        nn.init.xavier_normal_(self.emd)
        
    def forward(self, x):
        if(len(x.shape) == 2):
            # if input is onhot encoding(genre)
            x = x.type_as(self.emd)
            return t.mm(x, self.emd)
        else:
            return self.emd[x, :]
            
    
    
    
    
    
    
class SeqModel_pair(nn.Module):
    """Test combining direct embedding and sequence embedding(regarding sequence of rating as the feature of user/item)"""
    def __init__(self, input_size, hidden_size, num_ratings, num_layers, num_users, num_items, num_gender, num_age, num_occupation, num_zip, num_genre, sp_hidden_size, dropout):
        super(SeqModel_pair, self).__init__()
        
        self.user_extractor = Feature_Extractor(input_size = input_size, hidden_size = hidden_size, num_ratings = num_ratings, num_layers = num_layers, num_ui = num_items)
        self.item_extractor = Feature_Extractor(input_size = input_size, hidden_size = hidden_size, num_ratings = num_ratings, num_layers = num_layers, num_ui = num_users)
        # init sparse feature
        self.gender_emd = SparseFeature(total_num = num_gender, emd_size = sp_hidden_size)
        self.age_emd = SparseFeature(total_num = num_age, emd_size = sp_hidden_size)
        self.occupation_emd = SparseFeature(total_num = num_occupation, emd_size = sp_hidden_size)
        self.zip_emd = SparseFeature(total_num = num_zip, emd_size = sp_hidden_size)
        self.genre_emd = SparseFeature(total_num = num_genre, emd_size = sp_hidden_size)
        
        fc_inp_size = 2 * hidden_size + 5 * sp_hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(fc_inp_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss = nn.BCEWithLogitsLoss()

        
        
        self.drop = nn.Dropout(dropout)
    def forward(self, u_rating_seq, i_id_seq, i_rating_seq, u_id_seq, gender, age, occupation, zip_code, genre, labels):
        """
        u_rating_seq: user rating sequence
        i_id_seq: each rating related to a item id embedding
        i_rating_seq: item rating sequence
        u_id_seq: each rating related to a user id embedding
        gender: users' gender [0, 1]
        age: users' age
        occupation: users' occupation
        zip_code: zip code
        genre: movie's genre
        
        """

        u_emd = self.user_extractor(u_rating_seq, i_id_seq)

        i_emd = self.item_extractor(i_rating_seq, u_id_seq)

        gender_emd = self.gender_emd(gender.flatten())
        age_emd = self.age_emd(age.flatten())
        occupation_emd = self.occupation_emd(occupation.flatten())
        zipCode_emd = self.zip_emd(zip_code.flatten())
        genre_emd = self.genre_emd(genre)
        
        x_ui = t.cat((u_emd,i_emd, gender_emd, age_emd, occupation_emd, zipCode_emd, genre_emd), dim = 1)

        x_ui = self.fc(self.drop(x_ui))
        x_ui = x_ui.flatten()
#         x_uj = self.fc(self.drop(x_uj))
        labels = labels.flatten()
        labels = labels.type_as(x_ui)
        
        log_prob = self.loss(x_ui, labels)
        
        # no regularization
        return log_prob
    
class EmdGenerator(nn.Module):
    def __init__(self,inp_size, output_size):
        super(EmdGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(inp_size,512, bias = False),
            nn.Linear(512, output_size, bias = False),
            nn.Tanh(),
        )
    def forward(self, seq_emd):
        return self.generator(seq_emd)
    
    
    
class AggSeqModel_pair():
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_ratings,
                 num_layers,
                 num_users,
                 num_items,
                 num_gender,
                 num_age,
                 num_occupation,
                 num_zip,
                 num_genre,
                 sp_hidden_size,
                 seq_len,
                 dropout,
                 learning_rate,
                 epoch,
                 data_loader,
                 user_dict,
                 item_dict,
                 valid_df,
                 test_df,
                 name,
                 save = False,
                 load = False,
                ):
        self.model = SeqModel_pair(input_size = input_size, hidden_size = hidden_size, num_ratings = num_ratings,num_users = num_users, num_items = num_items, num_layers = num_layers,num_gender = num_gender,num_age = num_age, num_occupation = num_occupation, num_zip = num_zip, num_genre = num_genre, sp_hidden_size = sp_hidden_size, dropout = dropout).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.epoch = epoch
        self.data_loader = data_loader
        self.num_samples = len(data_loader)
        self.test_df = test_df
        self.valid_df = valid_df
        self.save = save
        self.name = name
        self.load = load
        self.hidden_size = hidden_size
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.seq_len = seq_len
        
    def predict_all(self, num_sample = 10):
        # random sample 5 on each side
        user_ids = self.test_df["user_id"].values
        item_ids = self.test_df["item_id"].values
        gender = self.test_df["gender"].values
        age = self.test_df["age"].values
        occupation = self.test_df["occupation"].values
        zipCode = self.test_df["zip"].values
        genres = self.test_df.iloc[:, 9:-1].values
        
        
        labels = self.test_df["label"].values
        pred_list = []
        for i in tqdm(range(len(user_ids))):
            list_uid_ratings = []
            list_uid_ids = []
            for _ in range(num_sample):
                u_index = np.random.choice(len(self.test_user_dict[user_ids[i]][0]))
                uid_rating_seq = self.test_user_dict[user_ids[i]][0][u_index]
                uid_id_seq = self.test_user_dict[user_ids[i]][1][u_index]
                list_uid_ratings.append(uid_rating_seq)
                list_uid_ids.append(uid_id_seq)
            
            list_iid_ratings = []
            list_iid_ids = []
            # for item emd
            for _ in range(num_sample):
                i_index = np.random.choice(len(self.test_item_dict[item_ids[i]][0]))
                iid_rating_seq = self.test_item_dict[item_ids[i]][0][i_index]
                iid_id_seq = self.test_item_dict[item_ids[i]][1][i_index]
                list_iid_ratings.append(iid_rating_seq)
                list_iid_ids.append(iid_id_seq)
                
            gender_in = np.repeat(gender[i], num_sample)
            age_in = np.repeat(age[i], num_sample)
            occupation_in = np.repeat(occupation[i], num_sample)
            zipCode_in = np.repeat(zipCode[i], num_sample)
            genre_in = np.repeat(genres[i, :][np.newaxis, ], num_sample, axis=0)
            
                
            uids_seq = t.LongTensor(list_uid_ratings).to(device)
            inp_user_ids = t.LongTensor(list_uid_ids).to(device)
            
            iids_seq = t.LongTensor(list_iid_ratings).to(device)
            inp_item_ids = t.LongTensor(list_iid_ids).to(device)
            
            gender_in = t.LongTensor(gender_in).to(device)
            age_in = t.LongTensor(age_in).to(device)
            occupation_in = t.LongTensor(occupation_in).to(device)
            zipCode_in = t.LongTensor(zipCode_in).to(device)
            genre_in = t.LongTensor(genre_in).to(device)
            
     
            self.model.eval()
            with t.no_grad():
                user_emd = self.model.user_extractor(uids_seq, inp_user_ids)
                item_emd = self.model.item_extractor(iids_seq, inp_item_ids)
                
                gender_emd = self.model.gender_emd(gender_in)
                age_emd = self.model.age_emd(age_in)
                occupation_emd = self.model.occupation_emd(occupation_in)
                zipCode_emd = self.model.zip_emd(zipCode_in)
                genre_emd = self.model.genre_emd(genre_in)

                x_ui = t.cat((user_emd,item_emd, gender_emd, age_emd, occupation_emd, zipCode_emd, genre_emd), dim = 1)
                
                pred = self.model.fc(x_ui)
                pred = pred.flatten()
                pred = t.sigmoid(pred)
#                 pred = t.mul(user_emd, item_emd).sum(dim = 1)
#                 pred = t.sigmoid(pred)
            self.model.train()
            
            pred_list.append(pred.detach().cpu().numpy())
        
        return pred_list
    
    def up_down_sampling(self, uidict, unaspect):
        tem_list_rating = []
        tem_list_id = []
        key_list = list(uidict.keys())
        value_list = [len(uidict[k]) for k in uidict]
        sum_val = sum(value_list)
        for i in range(len(key_list) -1):
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
    
    def predict(self, user_ids, item_ids, gender, age, occupation, zipCode, genres, batch=512):
        steps = int(np.ceil(len(user_ids) / batch))
        pred_list = []
        for i in tqdm(range(steps)):
            start_idx = i * batch
            end_idx = min((i + 1)* batch, len(user_ids))
            
            list_uid_ratings = []
            list_uid_ids = []
            # for user emd
            for u in user_ids[start_idx : end_idx]:
                uid_rating_seq, uid_id_seq = self.up_down_sampling(self.user_dict[u], "item_id")
                list_uid_ratings.append(uid_rating_seq)
                list_uid_ids.append(uid_id_seq)
            
            list_iid_ratings = []
            list_iid_ids = []
            # for item emd
            for i in item_ids[start_idx : end_idx]:
                iid_rating_seq, iid_id_seq = self.up_down_sampling(self.item_dict[i], "user_id")
                list_iid_ratings.append(iid_rating_seq)
                list_iid_ids.append(iid_id_seq)
                
            gender_in = gender[start_idx : end_idx]
            age_in = age[start_idx : end_idx]
            occupation_in = occupation[start_idx : end_idx]
            zipCode_in = zipCode[start_idx : end_idx]
            genre_in = genres[start_idx:end_idx, :]
            
                
            uids_seq = t.LongTensor(list_uid_ratings).to(device)
            inp_user_ids = t.LongTensor(list_uid_ids).to(device)
            
            iids_seq = t.LongTensor(list_iid_ratings).to(device)
            inp_item_ids = t.LongTensor(list_iid_ids).to(device)
            
            gender_in = t.LongTensor(gender_in).to(device)
            age_in = t.LongTensor(age_in).to(device)
            occupation_in = t.LongTensor(occupation_in).to(device)
            zipCode_in = t.LongTensor(zipCode_in).to(device)
            genre_in = t.LongTensor(genre_in).to(device)
            
     
            self.model.eval()
            with t.no_grad():
                user_emd = self.model.user_extractor(uids_seq, inp_user_ids)
                item_emd = self.model.item_extractor(iids_seq, inp_item_ids)
                
                gender_emd = self.model.gender_emd(gender_in)
                age_emd = self.model.age_emd(age_in)
                occupation_emd = self.model.occupation_emd(occupation_in)
                zipCode_emd = self.model.zip_emd(zipCode_in)
                genre_emd = self.model.genre_emd(genre_in)

                x_ui = t.cat((user_emd,item_emd, gender_emd, age_emd, occupation_emd, zipCode_emd, genre_emd), dim = 1)
                
                pred = self.model.fc(x_ui)
                pred = pred.flatten()
                pred = t.sigmoid(pred)
#                 pred = t.mul(user_emd, item_emd).sum(dim = 1)
#                 pred = t.sigmoid(pred)
            self.model.train()
            
            pred_list.extend(pred.detach().cpu().numpy())
        
        return pred_list
    
    def generator_predict(self, user_ids, item_ids, user_seq_dict, item_seq_dict, batch = 512):

        user_seq_emd = user_seq_dict["emd"]
        user_map = user_seq_dict["map_dict"]
        item_seq_emd = item_seq_dict["emd"]
        item_map = item_seq_dict["map_dict"]
        
        steps = int(np.ceil(len(user_ids) / batch))
        pred_list = []
        for i in tqdm(range(steps)):
            start_idx = i * batch
            end_idx = min((i + 1)* batch, len(user_ids))
            uids_seq = [user_seq_emd[user_map[i]] for i in user_ids[start_idx : end_idx]]
            iids_seq = [item_seq_emd[item_map[i]] for i in item_ids[start_idx : end_idx]]
            uids_seq_tensor = t.Tensor(uids_seq).to(device)
            iids_seq_tensor = t.Tensor(iids_seq).to(device)
            
            self.userEmd_gen.eval()
            self.itemEmd_gen.eval()
            self.model.eval()
            with t.no_grad():
                user_gen_emd = self.userEmd_gen(uids_seq_tensor)
                item_gen_emd = self.itemEmd_gen(iids_seq_tensor)
                user_emd = t.cat((user_gen_emd, uids_seq_tensor), dim = 1)
                item_emd = t.cat((item_gen_emd, iids_seq_tensor), dim = 1)
                x_ui = t.cat((user_emd, item_emd), dim = 1)
                pred = self.model.fc(x_ui)
                pred = pred.flatten()
                pred = t.sigmoid(pred)
            self.model.train()
            self.userEmd_gen.train()
            self.itemEmd_gen.train()
            pred_list.extend(pred.detach().cpu().numpy())
            
        return pred_list
        
    
    def save_model(self):
        if(isinstance(self.model, SeqModel_pair)):
            print(f"Model's state_dict")
            for param_tensor in self.model.state_dict():
                print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            
            print(f"Optimizer's state_dict:")
            print("param_groups", "\t", self.optimizer.state_dict()["param_groups"])
        
        print(f"************************** save model's state_dict **************************")
        pathlib.Path('model').mkdir(parents=True, exist_ok=True)
        model_file = "model/{}_m.pkl".format(self.name)
        optimizer_file = "model/{}_p.pkl".format(self.name)
        t.save(self.model.state_dict(), model_file)
#         t.save(self.optimizer.state_dict(), optimizer_file)
        
    def save_generator(self):
        # save user generator model
        pathlib.Path('model').mkdir(parents=True, exist_ok=True)
        if(isinstance(self.userEmd_gen,EmdGenerator)): 
            print(f"user generator Model's state_dict")
            for param_tensor in self.userEmd_gen.state_dict():
                print(param_tensor, "\t", self.userEmd_gen.state_dict()[param_tensor].size())
            user_gen_file = "model/{}_user_gen.pkl".format(self.name)
            t.save(self.userEmd_gen.state_dict(), user_gen_file)
            
        if(isinstance(self.itemEmd_gen, EmdGenerator)):
            print(f"item generator Model's state_dict")
            for param_tensor in self.itemEmd_gen.state_dict():
                print(param_tensor, "\t", self.itemEmd_gen.state_dict()[param_tensor].size())
            item_gen_file = "model/{}_item_gen.pkl".format(self.name)
            t.save(self.itemEmd_gen.state_dict(), item_gen_file)
        
                
                
        
    def load_model(self):
        print(f"************************** load pretrained model of {self.name} **************************")
        model_file = "model/{}_m.pkl".format(self.name)
        optimizer_file = "model/{}_p.pkl".format(self.name)
        self.model.load_state_dict(t.load(model_file))
#         self.optimizer.load_state_dict(t.load(optimizer_file))
        
    def save_emd(self):
        # load pretrained model first
        self.load_model()
        # save rating embedding and all user/item embedding
        pathlib.Path("embedding").mkdir(parents = True, exist_ok = True)
        # save embedding
        user_rating_emd_f = "embedding/{}_userRatingEmd.pkl".format(self.name)
        item_rating_emd_f = "embedding/{}_itemRatingEmd.pkl".format(self.name)
        user_seq_emd_f = "embedding/{}_userSeq.pkl".format(self.name)
        item_seq_emd_f = "embedding/{}_itemSeq.pkl".format(self.name)
        
        # user item embedding
        user_seq_emd, item_seq_emd = prepare_user_item_emd(self.model, self.user_dict, self.item_dict)
        # rating embedding
        
        rating_emd_user = self.model.user_extractor.ratings_emd.detach().cpu().numpy()
        rating_emd_item = self.model.item_extractor.ratings_emd.detach().cpu().numpy()
        
        print(f"************************** save user rating sequence dictionary **************************") 
        with open(user_seq_emd_f, "wb") as handle:
            pickle.dump(user_seq_emd, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print(f"************************** save item rating sequence dictionary **************************") 
        
        with open(item_seq_emd_f, "wb") as handle:
            pickle.dump(item_seq_emd, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
        print(f"************************** save user embedding **************************") 
        
        with open(user_rating_emd_f, "wb") as handle:
            pickle.dump(rating_emd_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"************************** save item embedding **************************") 
        with open(item_rating_emd_f, "wb") as handle:
            pickle.dump(rating_emd_item, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def get_emd(self):
        user_seq_emd_f = "embedding/{}_userSeq.pkl".format(self.name)
        item_seq_emd_f = "embedding/{}_itemSeq.pkl".format(self.name)
        user_emd_f = "embedding/{}_userEmd.pkl".format(self.name)
        item_emd_f = "embedding/{}_itemEmd.pkl".format(self.name)
        
        print(f"========================== load user rating sequence dictionary ==========================") 
        with open(user_seq_emd_f, "rb") as handle:
            user_seq_dict = pickle.load(handle)
        
        print(f"========================== load item rating sequence dictionary ==========================") 
        
        with open(item_seq_emd_f, "rb") as handle:
            item_seq_dict = pickle.load(handle)
            
        print(f"========================== load user embedding ==========================") 
        
        with open(user_emd_f, "rb") as handle:
            user_emd = pickle.load(handle)
            
        print(f"========================== load item embedding ==========================") 
        with open(item_emd_f, "rb") as handle:
            item_emd = pickle.load(handle)
            
        
        return user_emd, item_emd, user_seq_dict, item_seq_dict
            
    
    
    def eval_auc(self, test_df):
        user_ids = test_df["user_id"].values
        item_ids = test_df["item_id"].values
        gender = test_df["gender"].values
        age = test_df["age"].values
        occupation = test_df["occupation"].values
        zipCode = test_df["zip"].values
        genres = test_df.iloc[:, 9:-1].values
        
        
        labels = test_df["label"].values
        
        pred = self.predict(user_ids, item_ids, gender, age, occupation, zipCode, genres)
#         print(pred)
        pred = np.clip(pred,1e-6,1-1e-6)
        
        auc_score = roc_auc_score(labels, pred)
        lloss = log_loss(labels, pred)
        
        return auc_score, lloss
    
    
    def eval_gen_auc(self,test_df, user_seq_dict, item_seq_dict):
        user_ids = test_df["user_id"].values
        item_ids = test_df["item_id"].values
        labels = test_df["label"].values
        
        pred = self.generator_predict(user_ids, item_ids, user_seq_dict, item_seq_dict)
#         print(pred)
        pred = np.clip(pred,1e-6,1-1e-6)
        
        auc_score = roc_auc_score(labels, pred)
        lloss = log_loss(labels, pred)
        
        return auc_score, lloss
        

        
    def train(self):
        epoch_loss = 0
        for u_rating_seq, i_id_seq, i_rating_seq, u_id_seq, gender, age, occp, zcode, genres, labels in tqdm(self.data_loader):
            loss = self.model(u_rating_seq.to(device), i_id_seq.to(device), i_rating_seq.to(device), u_id_seq.to(device),  gender.to(device), age.to(device), occp.to(device), zcode.to(device), genres.to(device), labels.to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        return epoch_loss / self.num_samples
    
    
    def construct_dataloader(self, embedding, seq_emdding_dict, batch_size):
        dataset = EmdGen(embedding, seq_emdding_dict)
        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = 32)
        return dataloader
        
    
    
    def train_emd_gen(self, epoch):
        # prepare model
        self.userEmd_gen = EmdGenerator(self.hidden_size, self.hidden_size).to(device)
        self.itemEmd_gen = EmdGenerator(self.hidden_size, self.hidden_size).to(device)
        # dont know which one to use tbh
        user_optimizer = optim.Adam(self.userEmd_gen.parameters(), lr = 0.01)
        item_optimizer = optim.Adam(self.itemEmd_gen.parameters(), lr = 0.01)
        
        # load emd
        user_emd, item_emd, user_seq_dict, item_seq_dict = self.get_emd()
        
        # prepare dataloader
        userDataLoader = self.construct_dataloader(user_emd, user_seq_dict, batch_size = 56)
        itemDataLoader = self.construct_dataloader(item_emd, item_seq_dict, batch_size = 56)
        # train user emd generator
        # loss(mse)
        print(f"train user embedding generator  + item embedding generator====>")
        criterion = nn.MSELoss(reduction = "sum")

        for i in range(1, epoch + 1):
            user_epoch_loss = 0
            user_step = 0
            for user_emd, user_seq_emd in tqdm(userDataLoader):
                pred = self.userEmd_gen(user_seq_emd.to(device))
                loss = criterion(pred, user_emd.to(device))
                user_optimizer.zero_grad()
                loss.backward()
                user_optimizer.step()
                user_epoch_loss += loss
                user_step += 1
            print("===== user generator training: epoch %d / %d, loss: %.5f"%(i, epoch, user_epoch_loss / user_step))
            item_epoch_loss = 0
            item_step = 0
            for item_emd, item_seq_emd in tqdm(itemDataLoader):
                pred = self.itemEmd_gen(item_seq_emd.to(device))
                loss = criterion(pred, item_emd.to(device))
                item_optimizer.zero_grad()
                loss.backward()
                item_optimizer.step()
                item_epoch_loss += loss
                item_step += 1
            print("***** item generator training: epoch %d / %d, loss: %.5f"%(i, epoch, item_epoch_loss / item_step))
            
            print(f"<================== evaluation ==================>")
            auc_score, loss = self.eval_gen_auc(self.test_df, user_seq_dict, item_seq_dict)
            print(f"auc score on test: {auc_score} ======== logloss on test: {loss}")
            
        # save user/item generator
        print(f"save user/item embedding generator ====>")
        self.save_generator()

    
        
    def run(self):
        if(self.load):
            self.load_model()
        self.results = []
        for i in range(1, self.epoch + 1):
            epoch_loss = self.train()
            print("epoch: %d, loss: %.4f" %(i, epoch_loss))
            if(i % 1 == 0):
                print("===> evaluation")
                auc_score_v, loss_v = self.eval_auc(self.valid_df)
                auc_score_t, loss_t = self.eval_auc(self.test_df)
                self.results.append((auc_score_v, loss_v, auc_score_t, loss_t))
                print(f"auc score on valid: {auc_score_v} ======== logloss on valid: {loss_v}")
                print(f"auc score on test: {auc_score_t} ======== logloss on test: {loss_t}")
                
        if(self.save):
            self.save_model()

    
def recommend(u, mask, user_dict, item_dict, model):
    
    # finish the mask part
    model.eval()
    u = user_dict[u][np.random.choice(len(user_dict[u]))]
    i = [item_dict[iid][np.random.choice(len(item_dict[iid]))] for iid in item_dict]
    u = t.LongTensor(u).to(device)
    i = t.LongTensor(i).to(device)
    u = u.unsqueeze(0)
    
    with t.no_grad():
        # each u map all items 
        user_emd = model.user_extractor(u)
        all_item_emd = model.item_extractor(i)
        # expand user emd
        ep_user_emd = user_emd.expand(all_item_emd.shape[0], all_item_emd.shape[1])
        x_ui = t.cat((ep_user_emd, all_item_emd), dim = 1)
        out = model.fc(x_ui)
        out[mask,:] = t.Tensor([-float("inf")])
        pred = t.argsort(out, dim = 0)
    model.train()
    return pred


def prepare_user_item_emd(model, user_dict, item_dict):
    user_emd_dict = {}
    item_emd_dict = {}
    for u in tqdm(user_dict):
        uids_seq = t.LongTensor([user_dict[u][0]]).to(device)
        inp_user_ids = t.LongTensor([user_dict[u][1]]).to(device)
        model.eval()
        with t.no_grad():
            user_emd = model.user_extractor(uids_seq, inp_user_ids)
        model.train()
        user_emd_dict[u] = user_emd.detach().cpu().numpy()
    
    for i in tqdm(item_dict):
        iids_seq = t.LongTensor([item_dict[i][0]]).to(device)
        inp_item_ids = t.LongTensor([item_dict[i][1]]).to(device)
        model.eval()
        with t.no_grad():
            item_emd = model.item_extractor(iids_seq, inp_item_ids)
        model.train()
        item_emd_dict[i] = item_emd.detach().cpu().numpy()
        
    
    return user_emd_dict, item_emd_dict
    
##########################################################################################################################################################                

def auc_eval(model, user_dict, item_dict, train_user_list, test_user_list):
    user_emd, item_emd = prepare_user_item_emd(model, user_dict, item_dict)
    user_emd = user_emd.detach().cpu()
    item_emd = item_emd.detach().cpu()
    
            
def precision_and_recall_k(model, user_dict, item_dict, train_user_list, test_user_list, klist, batch=256):
    """Compute precision at k using GPU.
    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # get all user emd and item emd
    user_emb, item_emb, umap, imap = prepare_user_item_emd(model, user_dict, item_dict)
    user_emb = user_emb.detach().cpu()
    item_emb = item_emb.detach().cpu()
    
    # Calculate max k value
    max_k = max(klist)

    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            items_idx = train_user_list[i+j]
            if(items_idx != []):
                emd_index = [imap[i] for i in items_idx]
                mask[j].scatter_(dim=0, index=t.tensor(list(emd_index)), value=t.tensor(0.0))
        # Calculate prediction value
        
        cur_result = t.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = t.sigmoid(cur_result)
        assert not t.any(t.isnan(cur_result))
        # Make zero for already observed item
        cur_result = t.mul(mask, cur_result)
        _, cur_result = t.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else t.cat((result, cur_result), dim=0)

    result = result.cpu()
    # Sort indice and get test_pred_topk
    precisions, recalls = [], []
    for k in klist:
        precision, recall = 0, 0
        for i in range(user_emb.shape[0]):
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / max([min([k, len(test)]), 1])
            recall += val / max([len(test), 1])
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls
            
        
    
    
            