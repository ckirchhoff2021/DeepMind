import numpy as np
import pandas as pd
from common_path import *
from algorithms.fm import FMModel


class MovieDataset:
    def __init__(self):
        self.user_csv = os.path.join(mlen_path, 'users.csv')
        self.movie_csv = os.path.join(mlen_path, 'movies.csv')
        self.ration_csv = os.path.join(mlen_path, 'ratings.csv')
        self.user_dict = dict()
        self.movie_dict = dict()
        self.behaviors = dict()
        self.initialize()

    def initialize(self):
        user_df = pd.read_csv(self.user_csv)
        user_datas = user_df.to_numpy()
        print('==> user datas: ', len(user_datas))
        for data in user_datas:
            user_id, gender, age, occupation, code = data
            self.user_dict[user_id] = {
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'code': code
            }

        movide_df = pd.read_csv(self.movie_csv)
        movie_datas = movide_df.to_numpy()
        print('==> movie datas: ', len(movie_datas))
        for data in movie_datas:
            movie_id, title, genres = data
            self.movie_dict[movie_id] = {
                'title': title,
                'genres': genres
            }

        ration_df = pd.read_csv(self.ration_csv)
        rating_datas = ration_df.to_numpy()
        print('==> ration datas: ', len(rating_datas))
        for data in rating_datas:
            user_id, movie_id, rate, time_stamp = data
            if user_id not in self.behaviors.keys():
                self.behaviors[user_id] = list()
            self.behaviors[user_id].append(movie_id)

    def generate_datas(self):
        samples = list()
        items = set(self.movie_dict.keys())
        for user_id in self.behaviors.keys():
            pos_list = self.behaviors[user_id]
            neg_list = list(items - set(pos_list))
            for pos_id in pos_list:
                negs = np.random.choice(neg_list, 1)
                for neg_id in negs:
                    samples.append([user_id, pos_id, neg_id])
        print('==> all samples: ', len(samples))

        user_samples = dict()
        user_samples['gender'] = list()
        user_samples['age'] = list()
        user_samples['occupation'] = list()
        user_samples['code'] = list()

        positive_samples = dict()
        positive_samples['genres'] = list()

        negative_samples = dict()
        negative_samples['genres'] = list()
        
        np.random.shuffle(samples)
        samples = samples[:10000]

        for sample in samples:
            user_id, pos_id, neg_id = sample
            user_info = self.user_dict[user_id]
            user_samples['gender'].append(user_info['gender'])
            user_samples['age'].append(user_info['age'])
            user_samples['occupation'].append(user_info['occupation'])
            user_samples['code'].append(user_info['code'])

            pos_info = self.movie_dict[pos_id]
            pos_genres = pos_info['genres']
            genres_list = pos_genres.split('|')
            while len(genres_list) < 3:
                genres_list.append('-1024')
            positive_samples['genres'].append(genres_list[:3])

            neg_info = self.movie_dict[neg_id]
            neg_genres = neg_info['genres']
            genres_list = neg_genres.split('|')
            while len(genres_list) < 3:
                genres_list.append('-1024')
            negative_samples['genres'].append(genres_list[:3])

        return user_samples, positive_samples, negative_samples


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