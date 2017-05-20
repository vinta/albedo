# coding: utf-8

import argparse

from pyspark import SparkConf
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from albedo_toolkit.common import loadRawData
from albedo_toolkit.common import recommendItems
from albedo_toolkit.evaluators import RankingEvaluator
from albedo_toolkit.transformers import DataCleaner
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

rawDF = loadRawData()

# format data

ratingBuilder = RatingBuilder()
ratingDF = ratingBuilder.transform(rawDF)
ratingDF.cache()

# clean data

dataCleaner = DataCleaner(
    minItemStargazersCount=2,
    maxItemStargazersCount=4000,
    minUserStarredCount=2,
    maxUserStarredCount=5000
)
cleanDF = dataCleaner.transform(ratingDF)

# train model

wholeDF = cleanDF
wholeDF.cache()

als = ALS(implicitPrefs=True, seed=42) \
    .setRank(50) \
    .setRegParam(0.5) \
    .setAlpha(40) \
    .setMaxIter(22)

alsModel = als.fit(wholeDF)

# predict preferences

predictedDF = alsModel.transform(wholeDF)

predictionProcessor = PredictionProcessor()
predictionDF = predictionProcessor.transform(predictedDF)

# evaluate model

k = 30
rankingEvaluator = RankingEvaluator(k=k)
ndcg = rankingEvaluator.evaluate(predictionDF)
print('NDCG', ndcg)

# recommend items

username = args.username
recommendedItemsDF = recommendItems(rawDF, alsModel, username, topN=k, excludeKnownItems=False)
for item in recommendedItemsDF.collect():
    repoName = item['repo_full_name']
    repoUrl = 'https://github.com/{0}'.format(repoName)
    print(repoUrl, item['prediction'], item['repo_language'], item['stargazers_count'])

spark.stop()
