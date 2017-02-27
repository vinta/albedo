# coding: utf-8

import sqlite3

from django.core.management.base import BaseCommand

from graphlab import ranking_factorization_recommender
import graphlab as gl


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('-u', '--username', action='store', dest='username', required=True)

    def handle(self, *args, **options):
        active_user = options['username']

        print(self.style.SUCCESS('Active user: @{0}'.format(active_user)))

        conn = sqlite3.connect('db.sqlite3')
        sf = gl.SFrame.from_sql(conn, "SELECT from_username, repo_full_name, 1 AS 'rating' FROM app_repostarring;")

        training_data, validation_data = gl.recommender.util.random_split_by_user(sf, 'from_username', 'repo_full_name')
        model = ranking_factorization_recommender.create(
            training_data,
            user_id='from_username',
            item_id='repo_full_name',
            target='rating',
            binary_target=True
        )

        users = [active_user, ]
        recommends = model.recommend(users, k=50, diversity=1, exclude_known=True)
        for rec in recommends:
            print('{0} https://github.com/{1} / {2}'.format(rec['rank'], rec['repo_full_name'], rec['score']))
