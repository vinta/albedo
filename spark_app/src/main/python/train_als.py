# coding: utf-8

import argparse

from pyspark import SparkConf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

from albedo_toolkit.common import loadRawData
from albedo_toolkit.common import recommendItems
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

ratingBuilder = RatingBuilder(minStargazersCount=10000)
ratingDF = ratingBuilder.transform(rawDF)

popularItemsBuilder = PopularItemsBuilder(minStargazersCount=1000)
popularItemsDF = popularItemsBuilder.transform(rawDF)
popularItems = [row['item'] for row in popularItemsDF.select('item').collect()]
bcPopularItems = sc.broadcast(popularItems)

negativeGenerator = NegativeGenerator(negativePositiveRatio=20, bcPopularItems=bcPopularItems)
balancedDF = negativeGenerator.transform(ratingDF)

# train model

wholeDF = balancedDF
wholeDF.cache()

als = ALS(implicitPrefs=True, seed=42) \
    .setRank(50) \
    .setMaxIter(22) \
    .setRegParam(0.5) \
    .setAlpha(40)

alsModel = als.fit(wholeDF)
# alsModel.save('spark_persistence/alsModel')

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

username = args.username
recommendedItemsDF = recommendItems(rawDF, alsModel, username)
for item in recommendedItemsDF.collect():
    repoName = item['repo_full_name']
    repoUrl = 'https://github.com/{0}'.format(repoName)
    print(repoUrl, item['prediction'], item['repo_language'], item['stargazers_count'])

spark.stop()
