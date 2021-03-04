import numpy as np
import pandas as pd
from common_path import *
import tensorflow as tf


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
            rating_list.append(rate / 5.0)
        return sample_dict, rating_list


class FMRegressor:
    @staticmethod
    def get_variable(name, shape):
        variable = tf.get_variable(name, shape=shape, dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.002))
        return variable
    
    def __init__(self):
        self.feature_colnames = ['age', 'gender', 'occupation', 'code', 'genres']
        self.hash_buckets = [100, 10, 1000, 10000, 1000]
        self.feature_dims = [10, 10, 10, 10, 10]
        self.feature_types = [tf.int32, tf.string, tf.int32, tf.string, tf.string]
        self.feature_shapes = [[None], [None], [None], [None], [None, 3]]

        self.variable_dict = dict()
        self.placeholder_dict = dict()
        for i, colname in enumerate(self.feature_colnames):
            n_hash = self.hash_buckets[i]
            n_dim = self.feature_dims[i]
            variable = self.get_variable(colname, [n_hash, n_dim])
            self.variable_dict[colname] = variable
            self.placeholder_dict[colname] = tf.placeholder(self.feature_types[i], self.feature_shapes[i])

        self.placeholder_dict['label'] = tf.placeholder(tf.float32, [None])
        self.variable_dict['w1'] = self.get_variable('w1', [70, 128])
        self.variable_dict['b1'] = self.get_variable('b1', [128])
        self.variable_dict['w2'] = self.get_variable('w2', [128, 1])
        self.variable_dict['b2'] = self.get_variable('b2', [1])
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def get_embedding(self):
        embedding_list = list()
        for i, colname in enumerate(self.feature_colnames):
            n_hash = self.hash_buckets[i]
            variable = self.variable_dict[colname]
            placeholder = self.placeholder_dict[colname]
            feature_type = self.feature_types[i]
            if feature_type == tf.string:
                indices = tf.strings.to_hash_bucket_fast(placeholder, n_hash)
                embedding = tf.nn.embedding_lookup(variable, indices)
            else:
                embedding = tf.nn.embedding_lookup(variable, placeholder)
            embedding = tf.layers.flatten(embedding)
            embedding_list.append(embedding)
        embedding = tf.concat(embedding_list, 1)
        return embedding

    def inference(self):
        x = self.get_embedding()
        w1 = self.variable_dict['w1']
        b1 = self.variable_dict['b1']
        w2 = self.variable_dict['w2']
        b2 = self.variable_dict['b2']
        
        y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
        y2 = tf.squeeze(y2)
        return y2

    def get_feeding(self, sample_dict, labels, k1=-1, k2=-1):
        ret = dict()
        for colname in self.feature_colnames:
            placeholder = self.placeholder_dict[colname]
            datas = sample_dict[colname]
            slice = datas if k1 == -1 else datas[k1:k2]
            ret[placeholder] = slice
        placeholder = self.placeholder_dict['label']
        ret[placeholder] = labels if k1 == -1 else labels[k1:k2]
        return ret


    def fit(self, sample_dict, labels, epochs):
        print('All datas: ', len(labels))
        yp = self.inference()
        yt = self.placeholder_dict['label']
        loss = tf.losses.mean_squared_error(yt, yp)
        tf.summary.scalar('loss', loss)
        summary = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
        init_op = [ tf.global_variables_initializer(), tf.local_variables_initializer()]
        train_writer = tf.summary.FileWriter(os.path.join(output_path, 'regression'), self.sess.graph)

        self.sess.run(init_op)
        batch_size = 512
        batches = int(len(labels) / batch_size) + 1
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(batches):
                k1 = (i * batch_size)
                k2 = (i+1) * batch_size if (i+1) * batch_size < len(labels) else len(labels)
                feed = self.get_feeding(sample_dict, labels, k1, k2)
                vloss, _, summaries = self.sess.run([loss, optimizer, summary], feed_dict=feed)
                train_writer.add_summary(summaries, global_step=(epoch * batches + i))
                epoch_loss += vloss
                if i % 100 == 0:
                    print('==> Epoch:[%d]/[%d]-[%d][%d], loss=%f'%(epoch, epochs, i, batches, vloss))
            epoch_loss /= batches
            print('==> Epoch:[%d]/[%d], average loss=%f' % (epoch, epochs, epoch_loss))
            self.saver.save(self.sess, os.path.join(output_path, 'regression/regressor.ckpt'))


    def batch_fit(self, sample_dict, labels, epochs):
        print('All datas: ', len(labels))
        yp = self.inference()
        yt = self.placeholder_dict['label']
        loss = tf.losses.mean_squared_error(yt, yp)
        tf.summary.scalar('loss', loss)
        summary = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        train_writer = tf.summary.FileWriter(os.path.join(output_path, 'regression'), self.sess.graph)
        self.sess.run(init_op)
        batch_size = 512
        batches = int(len(labels) / batch_size) + 1

        dataset = tf.data.Dataset.from_tensor_slices((sample_dict, labels))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=len(labels)).batch(batch_size, drop_remainder=False).repeat(epochs)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()

        icount = 0
        epoch_loss = 0.0
        try:
            while True:
                icount += 1
                batch_sample, batch_label = self.sess.run(element)
                feed = self.get_feeding(batch_sample, batch_label)
                vloss, _, summaries = self.sess.run([loss, optimizer, summary], feed_dict=feed)
                train_writer.add_summary(summaries, global_step=icount)
                epoch_loss += vloss
                epoch = int(icount / batches)
                if icount % 100 == 0:
                    print('==> Epoch:[%d]/[%d]-[%d][%d], loss=%f' % (epoch, epochs, icount%batches, batches, vloss))
                if icount % batches == 0:
                    print('==> Epoch:[%d]/[%d], average loss=%f' % (epoch, epochs, epoch_loss/batches))
                    epoch_loss = 0.0
                    self.saver.save(self.sess, os.path.join(output_path, 'regression/regressor.ckpt'))
        except tf.errors.OutOfRangeError:
            print('end.')

def main():
    datas = RatingDataset()
    sample_dict, labels = datas.get_samples()
    net = FMRegressor()
    # net.fit(sample_dict, labels, 5)
    net.batch_fit(sample_dict, labels, 5)


if __name__ == '__main__':
    main()