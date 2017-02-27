# coding: utf-8

import re

from django.core.management.base import BaseCommand
from django.db import connection

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

from app.models import RepoStarring
from app.utils_repo import prepare_user_item_df


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('-i', '--repo_full_name', action='store', dest='repo_full_name', required=True)

    def handle(self, *args, **options):
        repo_full_name = options['repo_full_name']

        self.stdout.write(self.style.SUCCESS('Active item: @{0}'.format(repo_full_name)))

        min_stargazers_count = 500
        user_item_df = prepare_user_item_df(min_stargazers_count=min_stargazers_count)

        rs = RepoStarring.objects \
            .filter(stargazers_count__gte=min_stargazers_count) \
            .values_list('repo_full_name', 'repo_description', 'repo_language') \
            .distinct()

        query, params = rs.query.sql_with_params()
        df = pd.io.sql.read_sql_query(query, connection, params=params)

        tdf = pd.DataFrame()
        tdf['text'] = df['repo_full_name'].map(lambda text: text.replace('/', ' ')) + ' ' + df['repo_language'] + ' ' + df['repo_description']

        documents = tdf['text']

        class LemmaTokenizer(object):

            def __init__(self):
                self.tokenize = lambda doc: re.compile(r'(?u)\b\w\w+\b').findall(doc)
                self.lemmatize = WordNetLemmatizer().lemmatize

            def __call__(self, doc):
                return [self.lemmatize(token) for token in self.tokenize(doc)]

        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', ngram_range=(1, 2), min_df=2)
        tfidf_matrix = vectorizer.fit_transform(documents)
        print('tfidf_matrix', tfidf_matrix.shape)

        item_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        print('item_similarities', item_similarities.shape)

        print(self.style.SUCCESS('Recommended repositories:'))

        idx = df[df['repo_full_name'] == repo_full_name].index.get_values()[0]
        similar_indices = item_similarities[idx].argsort()[:-50:-1]
        similar_items = [(item_similarities[idx][i], df['repo_full_name'][i]) for i in similar_indices][1:]
        for i, (similarity, repo) in enumerate(similar_items):
            print(self.style.SUCCESS('{0:02d}. https://github.com/{1} / ({2})'.format(i + 1, repo, similarity)))
