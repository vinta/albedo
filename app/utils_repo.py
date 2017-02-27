# coding: utf-8

import logging
import pickle

import numpy as np
import pandas as pd

from app.models import RepoStarring

logger = logging.getLogger('django')


def prepare_user_item_df(min_stargazers_count):
    repos = RepoStarring.objects \
        .filter(stargazers_count__gte=min_stargazers_count) \
        .values_list('repo_full_name', flat=True) \
        .order_by('repo_full_name') \
        .distinct()
    repos_array = np.fromiter(repos.iterator(), np.dtype('U140'))
    repos_array.shape
    n_repos = repos_array.shape[0]

    users = RepoStarring.objects \
        .filter(stargazers_count__gte=min_stargazers_count) \
        .values_list('from_username', flat=True) \
        .order_by('from_username') \
        .distinct()
    users_array = np.fromiter(users.iterator(), np.dtype('U39'))
    users_array.shape
    n_users = users_array.shape[0]

    logger.info('Build the utility matrix')
    logger.info('The number of users: {0}'.format(n_users))
    logger.info('The number of items: {0}'.format(n_repos))

    filename = 'caches/df-{0}x{1}.pickle'.format(n_users, n_repos)
    try:
        user_item_df = pickle.load(open(filename, 'rb'))
    except IOError:
        shape = (n_users, n_repos)
        matrix = np.zeros(shape, dtype=np.int8)
        for i, username in enumerate(users_array):
            user_starred = RepoStarring.objects \
                .filter(from_username=username) \
                .values_list('repo_full_name', flat=True)
            user_starred_array = np.fromiter(user_starred.iterator(), np.dtype('U140'))
            row = np.in1d(repos_array, user_starred_array, assume_unique=True)
            matrix[i] = row.astype(np.float64)

        user_item_df = pd.DataFrame(matrix, columns=repos_array, index=users_array)
        user_item_df.to_pickle(filename)

    return user_item_df
