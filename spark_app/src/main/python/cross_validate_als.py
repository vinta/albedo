# coding: utf-8

from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession

from albedo_toolkit.common import loadRawData
from albedo_toolkit.common import printCrossValidationParameters
from albedo_toolkit.transformers import NegativeGenerator
from albedo_toolkit.transformers import OutputProcessor
from albedo_toolkit.transformers import PopularItemsBuilder
from albedo_toolkit.transformers import RatingBuilder


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

# cross-validate models

negativeGenerator = NegativeGenerator(bcPopularItems=bcPopularItems)

als = ALS(implicitPrefs=True, seed=42)

outputProcessor = OutputProcessor()

pipeline = Pipeline(stages=[
    negativeGenerator,
    als,
    outputProcessor,
])

paramGrid = ParamGridBuilder() \
    .addGrid(negativeGenerator.negativePositiveRatio, [10, 20]) \
    .addGrid(als.rank, [50, 60, 100]) \
    .addGrid(als.maxIter, [22, ]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(als.alpha, [1, 40, ]) \
    .build()

evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                          labelCol='rating',
                                          metricName='areaUnderROC')

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=2)

cvModel = cv.fit(ratingDF)

# evaluate models

printCrossValidationParameters(cvModel)
