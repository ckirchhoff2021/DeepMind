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
            samples.append([user_id, movie_id, rate])

    def get_samples(self):
        sample_dict = dict()
        sample_dict['age'] = list()
        sample_dict['gender'] = list()
        sample_dict['occupation'] = list()
        sample_dict['code'] = list()
        sample_dict['genres'] = list()
        rating_list = list()
        for index, (user_id, movie_id, rate) in self.samples:
            gender, age, occupation, code = self.user_dict[user_id]
            title, genres = self.movie_dict[movie_id]
            sample_dict['age'].append(age)
            sample_dict['gender'].append(gender)
            sample_dict['occupation'].append(occupation)
            sample_dict['code'].append(code)
            sample_dict['genres'].append(genres)
            rating_list.append(rate)
        return sample_dict


class FMRegressor:
    def __init__(self):
        self.feature_colnames = ['age', 'gender', 'occupation', 'code', 'genres']
        self.sess = tf.Session()
        self.saver = tf.train.Saver()



def main():
    dataset = MovieDataset()
    user_info, pos_info, neg_info = dataset.generate_datas()

    user_config = [
        {'name': 'gender', 'dim': 10, 'type': 'string', 'sequence': 1, 'buckets': 3},
        {'name': 'age', 'dim': 10, 'type': 'int', 'sequence': 1, 'buckets': 100},
        {'name': 'occupation', 'dim': 10, 'type': 'int', 'sequence': 1, 'buckets': 30},
        {'name': 'code', 'dim': 10, 'type': 'string', 'sequence': 1, 'buckets': 5000}
    ]

    item_config = [
        {'name': 'genres', 'dim': 10, 'type': 'string', 'sequence': 3, 'buckets': 1000}
    ]

    model = FMModel(user_config, item_config)
    model_dir = os.path.join(output_path, 'fm')
    summary_dir = os.path.join(output_path, 'summary')
    # model.train(user_info, pos_info, neg_info, model_dir, summary_dir)
    model.batch_train(user_info, pos_info, neg_info, model_dir, summary_dir, batch_size=128, epochs=10)
    # model.restore(model_dir)
    # embedding = model.predict(pos_info, 'positive')
    # print(embedding)


if __name__ == '__main__':
    main()