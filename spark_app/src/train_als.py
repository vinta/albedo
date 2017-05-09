# coding: utf-8

import sys

from pyspark import keyword_only
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml import Transformer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.param.shared import Param
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
import pyspark.sql.functions as F


active_user = str(sys.argv[1]) if len(sys.argv) > 1 else 'vinta'

conf = SparkConf()
# conf.set('spark.executor.memory', '8G')
# conf.set('spark.driver.memory', '8G')

spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .appName('albedo') \
    .getOrCreate()

sc = spark.sparkContext


class NegativeSampleGenerator(Transformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, negativePositiveRatio=1, bcPopularItems=()):
        super(NegativeSampleGenerator, self).__init__()
        self.negativePositiveRatio = Param(self, 'negativePositiveRatio', None)
        self.bcPopularItems = Param(self, 'bcPopularItems', None)
        self._setDefault(negativePositiveRatio=1)
        self._setDefault(bcPopularItems=())
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, negativePositiveRatio=1, bcPopularItems=()):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setNegativePositiveRatio(self, value):
        self._paramMap[self.negativePositiveRatio] = value
        return self

    def getNegativePositiveRatio(self):
        return self.getOrDefault(self.negativePositiveRatio)

    def setBCpopularItems(self, value):
        self._paramMap[self.bcPopularItems] = value
        return self

    def getBCpopularItems(self):
        return self.getOrDefault(self.bcPopularItems)

    def _transform(self, df):
        negativePositiveRatio = self.getNegativePositiveRatio()
        popularItems = self.getBCpopularItems().value

        def seqFunc(itemSet, item):
            itemSet.add(item)
            return itemSet

        def combFunc(itemSet1, itemSet2):
            return itemSet1.union(itemSet2)

        def getNegativeSamples(userItemsPair):
            user, positiveItems = userItemsPair
            positiveItemsCount = len(positiveItems)
            negativeItems = []
            negativeItemsCount = 0
            for popularItem in popularItems:
                if popularItem not in positiveItems:
                    negativeItems.append(popularItem)
                    negativeItemsCount += 1
                    if negativeItemsCount >= positiveItemsCount * negativePositiveRatio:
                        break
            return (user, negativeItems)

        def expandNegativeSamples(userItemsPair):
            user, negativeItems = userItemsPair
            return ((user, negative, 0) for negative in negativeItems)

        negativeRDD = df \
            .select('user', 'item') \
            .rdd \
            .aggregateByKey(set(), seqFunc, combFunc) \
            .map(lambda x: getNegativeSamples(x)) \
            .flatMap(lambda x: expandNegativeSamples(x))
        negativeDF = spark.createDataFrame(negativeRDD, ratingSchema)

        # avoid "Union can only be performed on tables with the same number of columns" in cross-validation
        columns = ('user', 'item', 'rating')
        balancedDF = df.select(*columns).union(negativeDF.select(*columns))
        return balancedDF


class OutputProcessor(Transformer, HasInputCol, HasOutputCol):

    def _transform(self, predictionDF):
        predictionDF = predictionDF.dropna(subset=['prediction', ])
        predictionDF = predictionDF.withColumn('prediction', predictionDF['prediction'].cast('double'))
        return predictionDF


# read data

# url = 'jdbc:mysql://127.0.0.1:3306/albedo'
# properties = {
#     'driver': 'com.mysql.jdbc.Driver',
#     'user': 'root',
#     'password': '123',
# }

url = 'jdbc:sqlite:file:///Users/vinta/Projects/albedo/db.sqlite3'
properties = {
    'driver': 'org.sqlite.JDBC',
    'date_string_format': 'yyyy-MM-dd HH:mm:ss'
}

rawDF = spark.read.jdbc(url, table='app_repostarring', properties=properties)
rawDF.cache()

# preprocess dataset

minStargazersCount = 0

ratingSchema = StructType([
    StructField('user', IntegerType(), nullable=False),
    StructField('item', IntegerType(), nullable=False),
    StructField('rating', IntegerType(), nullable=False),
])

ratingDF = rawDF \
    .selectExpr('from_user_id AS user', 'repo_id AS item', '1 AS rating') \
    .where('stargazers_count > {0}'.format(minStargazersCount)) \
    .orderBy('user')
ratingDF = spark.createDataFrame(ratingDF.rdd, ratingSchema)

popularItemsDF = rawDF \
    .where('stargazers_count > 1000') \
    .groupBy('repo_id') \
    .agg(F.max('stargazers_count').alias('stars')) \
    .orderBy('stars', ascending=False) \
    .withColumnRenamed('repo_id', 'item')

popularItems = [row['item'] for row in popularItemsDF.select('item').collect()]
bcPopularItems = sc.broadcast(popularItems)

# cross-validate models

negativeSampleGenerator = NegativeSampleGenerator(bcPopularItems=bcPopularItems)

als = ALS(implicitPrefs=True, seed=42)

outputProcessor = OutputProcessor()

pipeline = Pipeline(stages=[
    negativeSampleGenerator,
    als,
    outputProcessor,
])

paramGrid = ParamGridBuilder() \
    .addGrid(negativeSampleGenerator.negativePositiveRatio, [1, 2, 3, 4]) \
    .addGrid(als.rank, [50, 70]) \
    .addGrid(als.maxIter, [24, 26]) \
    .addGrid(als.regParam, [0.001, 0.01, 0.1, 0.5]) \
    .addGrid(als.alpha, [0.1, 1, 40]) \
    .build()

evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                          labelCol='rating',
                                          metricName='areaUnderROC')

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=2)

cvModel = cv.fit(ratingDF)

metric_params_pairs = list(zip(cvModel.avgMetrics, cvModel.getEstimatorParamMaps()))
metric_params_pairs.sort(key=lambda x: x[0], reverse=True)
best_metric_params = metric_params_pairs[0][1]
for pair in metric_params_pairs:
    metric, params = pair
    print('metric', metric)
    for k, v in params.items():
        print(k.name, v)
    print('\n')

# # train model

# negativeSampleGenerator = NegativeSampleGenerator(negativePositiveRatio=0, bcPopularItems=bcPopularItems)
# balancedDF = negativeSampleGenerator.transform(ratingDF)

# wholeDF = balancedDF
# wholeDF.cache()

# als = ALS(implicitPrefs=True, seed=42) \
#     .setRank(2) \
#     .setMaxIter(2) \
#     .setRegParam(0.1) \
#     .setAlpha(1)
# # .setRank(50) \
# # .setMaxIter(24) \
# # .setRegParam(0.1) \
# # .setAlpha(1)

# alsModel = als.fit(wholeDF)
# alsModel.save('spark_persistence/alsModel')

# # predict preferences

# outputProcessor = OutputProcessor()

# predictionDF = alsModel.transform(wholeDF)
# predictionDF = outputProcessor.transform(predictionDF)

# # evaluate model

# evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',
#                                           labelCol='rating',
#                                           metricName='areaUnderROC')
# aucROC = evaluator.evaluate(predictionDF)
# print('areaUnderROC', aucROC)

# # recommend items

# topN = 30
# username = active_user
# userID = rawDF \
#     .where('from_username = "{0}"'.format(username)) \
#     .select('from_user_id') \
#     .take(1)[0]['from_user_id']

# recommendItems = predictionDF \
#     .where('user = {0}'.format(userID)) \
#     .orderBy('prediction', ascending=False) \
#     .select('item', 'prediction') \
#     .limit(topN)

# repoDF = rawDF \
#     .groupBy('repo_id', 'repo_full_name', 'repo_language') \
#     .agg(F.max('stargazers_count').alias('stars')) \
#     .selectExpr('repo_id AS id', 'repo_full_name AS name', 'repo_language AS language', 'stars')

# recommendItemsWithInfo = recommendItems \
#     .join(repoDF, recommendItems['item'] == repoDF['id'], 'inner') \
#     .select('prediction', 'name', 'language', 'stars') \
#     .orderBy('prediction', ascending=False)

# for row in recommendItemsWithInfo.collect():
#     repoName = row['name']
#     repoUrl = 'https://github.com/{0}'.format(repoName)
#     print(repoUrl, row['prediction'], row['language'], row['stars'])

spark.stop()
