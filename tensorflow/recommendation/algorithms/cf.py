import math
import time
import pandas as pd
from common_path import *

class UserCF:
    def __init__(self):
        self.file_path = os.path.join(mlen_path, 'ratings.csv')
        self.frame = pd.read_csv(self.file_path)

    @staticmethod
    def __cosine_sim_(target_movies, movies):
        union_len = len(set(target_movies) & set(movies))
        if union_len == 0:
            return 0.0
        product = len(target_movies) * len(movies)
        cosine = union_len / math.sqrt(product)
        return cosine

    def _get_top_n_users_(self, target_user_id, top_n):
        '''
        calculate similarity between all users and return Top N similar users.
        '''
        target_movies = self.frame[self.frame['UserID'] == target_user_id]['MovieID']
        other_user_id = [i for i in set(self.frame['UserID']) if i != target_user_id]
        other_movies = [self.frame[self.frame['UserID'] == i]['MovieID'] for i in other_user_id]

        similarites = [self.__cosine_sim_(target_movies, movies) for movies in other_movies]
        similarites = sorted(zip(other_user_id, similarites), key=lambda x:x[1], reverse=True)
        return similarites[:top_n]


    def _get_candidates_items_(self, target_user_id):
        """
        Find all movies in source data and target_user did not meet before.
        """
        target_user_movies = set(self.frame[self.frame['UserID'] == target_user_id]['MovieID'])
        other_user_movies = set(self.frame[self.frame['UserID'] != target_user_id]['MovieID'])
        candidates_movies = list(target_user_movies ^ other_user_movies)
        return candidates_movies


    def _get_top_n_items_(self, top_n_users, candidates_movies, top_n):
        """
        calculate interest of candidates movies and return top n movies.
        e.g. interest = sum(sim * normalize_rating)
        """
        top_n_user_data = [self.frame[self.frame['UserID'] == k] for k, _ in top_n_users]
        interest_list = []
        for movie_id in candidates_movies:
            tmp = []
            for user_data in top_n_user_data:
                if movie_id in user_data['MovieID'].values:
                    tmp.append(user_data[user_data['MovieID'] == movie_id]['Rating'].values[0] / 5)
                else:
                    tmp.append(0)
            interest = sum([top_n_users[i][1] * tmp[i] for i in range(len(top_n_users))])
            interest_list.append((movie_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:top_n]



    def recommend(self, target_user_id=1, top_n=10):
        """
        user-cf for movies recommendation.
        """

        # most similar top n users
        top_n_users = self._get_top_n_users_(target_user_id, top_n)
        # candidates movies for recommendation
        candidates_movies = self._get_candidates_items_(target_user_id)
        # most interest top n movies
        top_n_movies = self._get_top_n_items_(top_n_users, candidates_movies, top_n)
        return top_n_movies



if __name__ == '__main__':
    start = time.time()
    cf = UserCF()
    movies = cf.recommend()
    for movie in movies:
        print(movie)
    print('Cost time: %f' % (time.time() - start))

