import numpy as np
import pandas as pd
from common_path import *
import tensorflow as tf

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class RatingDataset:
    def __init__(self):
        self.user_csv = os.path.join(mlen_path, 'users.csv')
        self.movie_csv = os.path.join(mlen_path, 'movies.csv')
        self.rating_csv = os.path.join(mlen_path, 'ratings.csv')
        self.user_dict = dict()
        self.movie_dict = dict()
        self.samples = list()
        self.initialize()

    def initialize(self):
        user_df = pd.read_csv(self.user_csv)
        user_datas = user_df.to_numpy()
        for data in user_datas:
            user_id, gender, age, occupation, code = data
            self.user_dict[user_id] = [gender, age, occupation, code]

        movide_df = pd.read_csv(self.movie_csv)
        movie_datas = movide_df.to_numpy()
        for data in movie_datas:
            movie_id, title, genres = data
            self.movie_dict[movie_id] = [title, genres]

        rating_df = pd.read_csv(self.rating_csv)
        rating_datas = rating_df.to_numpy()

        for data in rating_datas:
            user_id, movie_id, rate, time_stamp = data
            self.samples.append([user_id, movie_id, rate])

    def get_samples(self):
        sample_dict = dict()
        sample_dict['age'] = list()
        sample_dict['gender'] = list()
        sample_dict['occupation'] = list()
        sample_dict['code'] = list()
        sample_dict['genres'] = list()
        rating_list = list()
        for index, (user_id, movie_id, rate) in enumerate(self.samples):
            gender, age, occupation, code = self.user_dict[user_id]
            title, genres = self.movie_dict[movie_id]
            genres_list = genres.split('|')
            while len(genres_list) < 3:
                genres_list.append('-1024')
            sample_dict['age'].append(age)
            sample_dict['gender'].append(gender)
            sample_dict['occupation'].append(occupation)
            sample_dict['code'].append(code)
            sample_dict['genres'].append(genres_list[:3])
            rating_list.append(rate/5.0)
        return sample_dict, rating_list


class FMRegressor:
    @staticmethod
    def get_variable(name, shape):
        variable = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.2))
        return variable

    def __init__(self):
        self.feature_colnames = ['age', 'gender', 'occupation', 'code', 'genres']
        self.hash_buckets = [100, 10, 1000, 10000, 100]
        self.feature_types = [tf.int32, tf.string, tf.int32, tf.string, tf.string, tf.string]
        self.feature_shapes = [[None], [None], [None], [None], [None, 3]]

        self.variable_dict = dict()
        self.placeholder_dict = dict()
        for i, colname in enumerate(self.feature_colnames):
            self.variable_dict[colname] = self.get_variable(colname, [self.hash_buckets[i], 10])
            self.placeholder_dict[colname] = tf.placeholder(self.feature_types[i], shape=self.feature_shapes[i])

        self.variable_dict['w1'] = self.get_variable('w1', [70, 128])
        self.variable_dict['b1'] = self.get_variable('b1', 128)
        self.variable_dict['w2'] = self.get_variable('w2', [128, 1])
        self.variable_dict['b2'] = self.get_variable('b2', 1)
        self.placeholder_dict['score'] = tf.placeholder(tf.float32, shape=[None])

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def get_vector(self):
        embedding_list = list()
        for i, colname in enumerate(self.feature_colnames):
            placeholder = self.placeholder_dict[colname]
            variables = self.variable_dict[colname]
            feature_type = self.feature_types[i]
            if feature_type == tf.string:
                indices = tf.strings.to_hash_bucket_fast(placeholder, self.hash_buckets[i])
                embedding = tf.nn.embedding_lookup(variables, indices)
            else:
                indices = tf.cast(placeholder, tf.int32)
                embedding = tf.nn.embedding_lookup(variables, indices)
            embedding = tf.layers.flatten(embedding)
            embedding_list.append(embedding)
        embedding = tf.concat(embedding_list, 1)
        return embedding

    def get_feed_dict(self, sample_dict, labels, k1, k2):
        feed_dict = dict()
        for colname in self.feature_colnames:
            placeholder = self.placeholder_dict[colname]
            datas = sample_dict[colname]
            slices = datas[k1:k2]
            feed_dict[placeholder] = slices
        feed_dict[self.placeholder_dict['score']] = labels[k1:k2]
        return feed_dict

    def fit(self, sample_dict, labels, epochs):
        print('All datas: ', len(labels))
        batch_size = 128
        x = self.get_vector()
        w1 = self.variable_dict['w1']
        k1 = tf.matmul(x, w1)
        y1 = tf.matmul(x, self.variable_dict['w1']) + self.variable_dict['b1']
        y1 = tf.nn.relu(y1)
        y2 = tf.matmul(y1, self.variable_dict['w2']) + self.variable_dict['b2']
        y2 = tf.nn.sigmoid(y2)
        y2 = tf.squeeze(y2)
        yt = self.placeholder_dict['score']
        loss = tf.losses.mean_squared_error(yt, y2)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.sess.run(init_op)

        for epoch in range(epochs):
            batches = int(len(labels) / batch_size) + 1
            epoch_loss = 0.0
            for i in range(batches):
                k1 = i * batch_size
                k2 = len(labels) if (i + 1) * batch_size > len(labels) else (i + 1) * batch_size
                feed = self.get_feed_dict(sample_dict, labels, k1, k2)
                vloss, _ = self.sess.run([loss, optimizer], feed_dict=feed)
                epoch_loss += vloss
                if i % 50 == 0:
                    print('Epoch:[%d]/[%d]-[%d]/[%d], loss= %f' %(epoch, epochs, i, batches, vloss))
            print('Epoch:[%d]/[%d], average loss = %f' %(epoch, epochs, epoch_loss/batches))
            self.saver.save(self.sess, os.path.join(output_path, 'regression/regressor.ckpt'))



def main():
    net = FMRegressor()
    datas = RatingDataset()
    sample_dict, labels = datas.get_samples()
    net.fit(sample_dict, labels, 3)


if __name__ == '__main__':
    main()