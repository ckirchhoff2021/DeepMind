import math
import pickle
import random
import numpy as np
import pandas as pd
from common_path import *

'''
latent factor model 隐语义模型
'''

class Corpus:
    def __init__(self):
        self.file_path = os.path.join(mlen_path, 'ratings.csv')
        self.dict_file = os.path.join(output_path, 'lfm_item.pkl')
        self.frame = pd.read_csv(self.file_path)
        self.user_ids = set(self.frame['UserID'].values)
        self.item_ids = set(self.frame['MovieID'].values)
        self.items_dict = {user_id: self._get_pos_neg_item(user_id) for user_id in list(self.user_ids)}

    def _get_pos_neg_item(self, user_id):
        print('==> %s' % user_id)
        pos_item_ids = set(self.frame[self.frame['UserID'] == user_id]['MovieID'])
        neg_item_ids = self.item_ids - pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict

    def save(self):
        with open(self.dict_file, 'wb') as f:
            pickle.dump(self.items_dict, f)

    @staticmethod
    def load():
        f = open(os.path.join(output_path, 'lfm_item.pkl'), 'rb')
        items_dict = pickle.load(f)
        f.close()
        return items_dict

class LFM:
    def __init__(self):
        self.latent_dim = 5
        self.epochs = 5
        self.lr = 0.02
        self.lam = 0.01
        self.file_path = os.path.join(mlen_path, 'ratings.csv')
        self.frame = pd.read_csv(self.file_path)
        self.user_ids = set(self.frame['UserID'].values)
        self.item_ids = set(self.frame['MovieID'].values)
        self.items_dict = Corpus.load()

        array_p = np.random.randn(len(self.user_ids), self.latent_dim)
        array_q = np.random.randn(len(self.item_ids), self.latent_dim)
        self.p = pd.DataFrame(array_p, columns=range(0, self.latent_dim), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.latent_dim), index=list(self.item_ids))

    def _predict_(self, user_id, item_id):
        p = np.mat(self.p.loc[user_id].values)
        q = np.mat(self.q.loc[item_id].values).T
        r = (p * q).sum()
        logit = 1.0 / (1 + np.exp(-r))
        return logit

    def _err_(self, user_id, item_id, y):
        e = y - self._predict_(user_id, item_id)
        return e

    def _optimize_(self, user_id, item_id, e):
        gradient_p = -e * self.q.loc[item_id].values
        l2_p = self.lam * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.loc[user_id].values
        l2_q = self.lam * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        for step in range(self.epochs):
            for iu, (user_id, item_dict) in enumerate(self.items_dict.items()):
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for index, item_id in enumerate(item_ids):
                    loss = self._err_(user_id, item_id, item_dict[item_id])
                    self._optimize_(user_id, item_id, loss)
                    if index % 100 == 0:
                        print("==> Epoch:[%d]/[%d],user:[%d]/[%d], item:[%d]/[%d], loss = %f" %
                              (step, self.epochs, iu, len(self.user_ids), index, len(item_ids), loss))
            self.lr *= 0.9
            self.save()

    def save(self):
        f = open(os.path.join(model_path, 'lfm.model'), 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def predict(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.frame[self.frame['UserID']==user_id]['MovieID'])
        other_item_ids = self.item_ids & user_item_ids
        scores = [self._predict_(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), scores), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def load(self):
        f = open(os.path.join(model_path, 'lfm.model'), 'rb')
        self.p, self.q = pickle.load(f)
        f.close()




if __name__ == '__main__':
    # datas = Corpus()
    # datas.save()
    lfm = LFM()
    lfm.train()