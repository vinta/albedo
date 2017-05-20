# coding: utf-8

from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession

from albedo_toolkit.common import loadRawData
from albedo_toolkit.common import printCrossValidationParameters
from albedo_toolkit.evaluators import RankingEvaluator
from albedo_toolkit.transformers import DataCleaner
from albedo_toolkit.transformers import PredictionProcessor
from albedo_toolkit.transformers import RatingBuilder


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

# cross-validate models

dataCleaner = DataCleaner()

als = ALS(implicitPrefs=True, seed=42)

predictionProcessor = PredictionProcessor()

pipeline = Pipeline(stages=[
    dataCleaner,
    als,
    predictionProcessor,
])

paramGrid = ParamGridBuilder() \
    .addGrid(dataCleaner.minItemStargazersCount, [1, 10, 100]) \
    .addGrid(dataCleaner.maxItemStargazersCount, [4000, ]) \
    .addGrid(dataCleaner.minUserStarredCount, [1, 10, 100]) \
    .addGrid(dataCleaner.maxUserStarredCount, [1000, 4000, ]) \
    .addGrid(als.rank, [50, 100]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(als.alpha, [0.01, 0.89, 1, 40, ]) \
    .addGrid(als.maxIter, [22, ]) \
    .build()

rankingEvaluator = RankingEvaluator(k=30)

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=rankingEvaluator,
                    numFolds=2)

cvModel = cv.fit(ratingDF)

# show results

printCrossValidationParameters(cvModel)
