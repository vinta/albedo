# coding: utf-8

from django.core.management.base import BaseCommand

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd

from app.utils_repo import prepare_user_item_df


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('-u', '--username', action='store', dest='username', required=True)

    def handle(self, *args, **options):
        active_user = options['username']

        print(self.style.SUCCESS('Active user: @{0}'.format(active_user)))

        user_item_df = prepare_user_item_df(min_stargazers_count=100)
        n_users, n_items = user_item_df.shape

        print(self.style.SUCCESS('Build the utility matrix:'))
        print(self.style.SUCCESS('The number of users: {0}'.format(n_users)))
        print(self.style.SUCCESS('The number of items: {0}'.format(n_items)))

        print(self.style.SUCCESS('Calculate similarities of users'))

        user_item_matrix = user_item_df.as_matrix()
        similarity_method = 'dice'
        filename = 'caches/user-similarities-{0}x{1}-{2}.pickle'.format(n_users, n_items, similarity_method)
        try:
            user_similarities = np.load(open(filename, 'rb'))
        except IOError:
            user_similarities = 1 - pairwise_distances(user_item_matrix, metric=similarity_method)
            np.save(open(filename, 'wb'), user_similarities)

        print(self.style.SUCCESS('Calculate predictions'))

        users_array = user_item_df.index
        items_array = user_item_df.columns
        predictions = user_similarities.dot(user_item_matrix) / np.abs(user_similarities).sum(axis=1).T
        prediction_df = pd.DataFrame(predictions, index=users_array, columns=items_array)

        print('user_item_matrix: {0}'.format(user_item_matrix.shape))
        print('user_similarities: {0}'.format(user_similarities.shape))
        print('predictions: {0}'.format(predictions.shape))

        user_starred = user_item_df.loc[active_user, :][user_item_df.loc[active_user, :] == 1]
        user_unstarred = prediction_df.loc[active_user, :].drop(user_starred.index)
        user_unstarred.sort_values(ascending=False)
        recommendations = user_unstarred.sort_values(ascending=False).index.tolist()

        print(self.style.SUCCESS('Recommended repositories:'))

        for i, repo in enumerate(recommendations[:100]):
            print(self.style.SUCCESS('{0:02d}. https://github.com/{1}'.format(i + 1, repo)))
