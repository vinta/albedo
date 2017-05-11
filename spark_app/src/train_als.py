# coding: utf-8

import argparse

from pyspark import SparkConf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from albedo_toolkit.common import loadRawData
from albedo_toolkit.transformers import NegativeGenerator
from albedo_toolkit.transformers import OutputProcessor
from albedo_toolkit.transformers import PopularItemsBuilder
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
rawDF.cache()

# preprocess data

ratingBuilder = RatingBuilder(minStargazersCount=100)
ratingDF = ratingBuilder.transform(rawDF)

popularItemsBuilder = PopularItemsBuilder(minStargazersCount=1000)
popularItemsDF = popularItemsBuilder.transform(rawDF)
popularItems = [row['item'] for row in popularItemsDF.select('item').collect()]
bcPopularItems = sc.broadcast(popularItems)

negativeGenerator = NegativeGenerator(negativePositiveRatio=2, bcPopularItems=bcPopularItems)
balancedDF = negativeGenerator.transform(ratingDF)

# train model

wholeDF = balancedDF
wholeDF.cache()

als = ALS(implicitPrefs=True, seed=42) \
    .setRank(50) \
    .setMaxIter(22) \
    .setRegParam(0.1) \
    .setAlpha(1)

alsModel = als.fit(wholeDF)
alsModel.save('spark_persistence/alsModel')

# predict preferences

outputProcessor = OutputProcessor()

predictedDF = alsModel.transform(wholeDF)
outputDF = outputProcessor.transform(predictedDF)

# evaluate model

evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                          labelCol='rating',
                                          metricName='areaUnderROC')
areaUnderROC = evaluator.evaluate(outputDF)
print('areaUnderROC', areaUnderROC)

# recommend items

topN = 30
username = args.username
userID = rawDF \
    .where('from_username = "{0}"'.format(username)) \
    .select('from_user_id') \
    .take(1)[0]['from_user_id']

recommendItems = outputDF \
    .where('user = {0}'.format(userID)) \
    .orderBy('prediction', ascending=False) \
    .select('item', 'prediction') \
    .limit(topN)

repoDF = rawDF \
    .groupBy('repo_id', 'repo_full_name', 'repo_language') \
    .agg(F.max('stargazers_count').alias('stars')) \
    .selectExpr('repo_id AS id', 'repo_full_name AS name', 'repo_language AS language', 'stars')

recommendItemsWithInfo = recommendItems \
    .join(repoDF, recommendItems['item'] == repoDF['id'], 'inner') \
    .select('prediction', 'name', 'language', 'stars') \
    .orderBy('prediction', ascending=False)

for row in recommendItemsWithInfo.collect():
    repoName = row['name']
    repoUrl = 'https://github.com/{0}'.format(repoName)
    print(repoUrl, row['prediction'], row['language'], row['stars'])

spark.stop()
