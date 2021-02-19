import os
import json
import pickle
import pandas as pd
from common_path import *

'''
personal rank
'''

class Graph:
    def __init__(self):
        self.file_path = os.path.join(mlen_path, 'ratings.csv')
        self.frame = pd.read_csv(self.file_path)
        self.user_ids = list(set(self.frame['UserID']))
        self.item_ids = list(set(self.frame['MovieID']))
        self.graph = dict()
        self.process()

    def process(self):
        for user_id in self.user_ids:
            positives = list(set(self.frame[self.frame['UserID'] == user_id]['MovieID']))
            ikey = 'user_{}'.format(user_id)
            ivalue = list()
            for item_id in positives:
                ivalue.append('item_{}'.format(item_id))
            self.graph[ikey] = ivalue

        for item_id in self.item_ids:
            positives = list(set(self.frame[self.frame['MovieID'] == item_id]['UserID']))
            ikey = 'item_{}'.format(item_id)
            ivalue = list()
            for user_id in positives:
                ivalue.append('user_{}'.format(user_id))
            self.graph[ikey] = ivalue

    def save(self):
        with open(os.path.join(mlen_path, 'graph.json'), 'w') as f:
            json.dump(self.graph, f, ensure_ascii=False)

    @staticmethod
    def load():
        graph = json.load(open(os.path.join(mlen_path, 'graph.json'), 'r'))
        return graph


class PersonalRank:
    def __init__(self):
        self.graph = Graph.load()
        self.alpha = 0.6
        self.epochs = 20
        self.frame = pd.read_csv(os.path.join(mlen_path, 'ratings.csv'))
        self.params = {k: 0 for k in self.graph.keys()}

    def train(self, user_id):
        self.params['user_{}'.format(user_id)] = 1
        for epoch in range(self.epochs):
            print('Step {}...'.format(epoch))
            tmp = {k: 0 for k in self.graph.keys()}
            for node in self.graph.keys():
                edges = self.graph[node]
                for next_node in edges:
                    tmp[next_node] += self.alpha * self.params[node] / len(edges)
            tmp['user_' + str(user_id)] += 1 - self.alpha
            self.params = tmp
        self.params = sorted(self.params.items(), key=lambda x: x[1], reverse=True)

    def predict(self, user_id, top_n):
        self.train(user_id)
        item_ids = ['item_' + str(item_id) for item_id in list(set(self.frame[self.frame['UserID'] == user_id]['MovieID']))]
        # candidates = [(key, value) for key, value in self.params if key not in item_ids and 'user' not in key]
        predicts = [key for key, value in self.params if 'user' not in key]
        candidates = predicts[:top_n]
        recall = len(set(item_ids) & set(candidates)) / len(set(item_ids) | set(candidates))
        print(recall)
        return candidates[:top_n]


def main():
    g = Graph()
    g.save()

    pr = PersonalRank()
    candidates = pr.predict(1, 10)
    print(candidates)


if __name__ == '__main__':
    main()





