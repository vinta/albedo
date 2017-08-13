# coding: utf-8

import argparse

from pyspark import SparkConf
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from albedo_toolkit.common import load_raw_data
from albedo_toolkit.common import recommend_items
from albedo_toolkit.evaluators import RankingEvaluator
# from albedo_toolkit.transformers import DataCleaner
from albedo_toolkit.transformers import PredictionProcessor
from albedo_toolkit.transformers import RatingBuilder


parser = argparse.ArgumentParser()
parser.add_argument('-u', '--username', action='store', dest='username', required=True)
args = parser.parse_args()

conf = SparkConf()

spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()

sc = spark.sparkContext

# load data

raw_df = load_raw_data()

# format data

rating_builder = RatingBuilder()
rating_df = rating_builder.transform(raw_df)
rating_df.cache()

# clean data

# data_cleaner = DataCleaner(
#     min_item_stargazers_count=2,
#     max_item_stargazers_count=4000,
#     min_user_starred_count=2,
#     max_user_starred_count=5000
# )
# clean_df = data_cleaner.transform(rating_df)

# train model

whole_df = rating_df
whole_df.cache()

als = ALS(implicitPrefs=True, seed=42) \
    .setRank(50) \
    .setRegParam(0.5) \
    .setAlpha(40) \
    .setMaxIter(22)

als_model = als.fit(whole_df)

# predict preferences

predicted_df = als_model.transform(whole_df)

prediction_processor = PredictionProcessor()
prediction_df = prediction_processor.transform(predicted_df)

# evaluate model

k = 30
rankingEvaluator = RankingEvaluator(k=k)
ndcg = rankingEvaluator.evaluate(prediction_df)
print('NDCG', ndcg)

# recommend items

username = args.username
recommended_items_df = recommend_items(raw_df, als_model, username, top_n=k, exclude_known_items=False)
for item in recommended_items_df.collect():
    repo_name = item['repo_full_name']
    repo_url = 'https://github.com/{0}'.format(repo_name)
    print(repo_url, item['prediction'])

spark.stop()
