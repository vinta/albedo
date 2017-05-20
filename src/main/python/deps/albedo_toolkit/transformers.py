# coding: utf-8

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import expr
import pyspark.sql.functions as F


spark = SparkSession.builder.getOrCreate()


class RatingBuilder(Transformer):

    def _transform(self, rawDF):
        ratingDF = rawDF \
            .selectExpr('from_user_id AS user', 'repo_id AS item', '1 AS rating', 'starred_at') \
            .orderBy('user', F.col('starred_at').desc())
        return ratingDF


class DataCleaner(Transformer):

    @keyword_only
    def __init__(self, minItemStargazersCount=None, maxItemStargazersCount=None, minUserStarredCount=None, maxUserStarredCount=None):
        super(DataCleaner, self).__init__()
        self.minItemStargazersCount = Param(self, 'minItemStargazersCount', '移除 stargazer 數低於這個數字的 item')
        self.maxItemStargazersCount = Param(self, 'maxItemStargazersCount', '移除 stargazer 數超過這個數字的 item')
        self.minUserStarredCount = Param(self, 'minUserStarredCount', '移除 starred repo 數低於這個數字的 user')
        self.maxUserStarredCount = Param(self, 'maxUserStarredCount', '移除 starred repo 數超過這個數字的 user')
        self._setDefault(minItemStargazersCount=1, maxItemStargazersCount=50000, minUserStarredCount=1, maxUserStarredCount=50000)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, minItemStargazersCount=None, maxItemStargazersCount=None, minUserStarredCount=None, maxUserStarredCount=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setMinItemStargazersCount(self, value):
        self._paramMap[self.minItemStargazersCount] = value
        return self

    def getMinItemStargazersCount(self):
        return self.getOrDefault(self.minItemStargazersCount)

    def setMaxItemStargazersCount(self, value):
        self._paramMap[self.maxItemStargazersCount] = value
        return self

    def getMaxItemStargazersCount(self):
        return self.getOrDefault(self.maxItemStargazersCount)

    def setMinUserStarredCount(self, value):
        self._paramMap[self.minUserStarredCount] = value
        return self

    def getMinUserStarredCount(self):
        return self.getOrDefault(self.minUserStarredCount)

    def setMaxUserStarredCount(self, value):
        self._paramMap[self.maxUserStarredCount] = value
        return self

    def getMaxUserStarredCount(self):
        return self.getOrDefault(self.maxUserStarredCount)

    def _transform(self, ratingDF):
        minItemStargazersCount = self.getMinItemStargazersCount()
        maxItemStargazersCount = self.getMaxItemStargazersCount()
        minUserStarredCount = self.getMinUserStarredCount()
        maxUserStarredCount = self.getMaxUserStarredCount()

        toKeepItemsDF = ratingDF \
            .groupBy('item') \
            .agg(F.count('user').alias('stargazers_count')) \
            .where('stargazers_count >= {0} AND stargazers_count <= {1}'.format(minItemStargazersCount, maxItemStargazersCount)) \
            .orderBy('stargazers_count', ascending=False) \
            .select('item', 'stargazers_count')
        temp1DF = ratingDF.join(toKeepItemsDF, 'item', 'inner')

        toKeepUsersDF = temp1DF \
            .groupBy('user') \
            .agg(F.count('item').alias('starred_count')) \
            .where('starred_count >= {0} AND starred_count <= {1}'.format(minUserStarredCount, maxUserStarredCount)) \
            .orderBy('starred_count', ascending=False) \
            .select('user', 'starred_count')
        temp2DF = temp1DF.join(toKeepUsersDF, 'user', 'inner')

        cleanDF = temp2DF.select('user', 'item', 'rating', 'starred_at')
        return cleanDF


class PredictionProcessor(Transformer):

    def _transform(self, predictedDF):
        nonNullDF = predictedDF.dropna(subset=['prediction', ])
        predictionDF = nonNullDF.withColumn('prediction', nonNullDF['prediction'].cast('double'))
        return predictionDF


# DEPRECATE
class PopularItemsBuilder(Transformer):

    def _transform(self, rawDF):
        popularItemsDF = rawDF \
            .where('stargazers_count > 1000') \
            .groupBy('repo_id') \
            .agg(F.max('stargazers_count').alias('stars')) \
            .orderBy('stars', ascending=False) \
            .withColumnRenamed('repo_id', 'item')
        return popularItemsDF


# DEPRECATE
class NegativeGenerator(Transformer):

    @keyword_only
    def __init__(self, negativeRatingValue=None, negativePositiveRatio=None, bcPopularItems=None):
        super(NegativeGenerator, self).__init__()
        self.negativeRatingValue = Param(self, 'negativeRatingValue', '負樣本的 rating 值')
        self.negativePositiveRatio = Param(self, 'negativePositiveRatio', '負樣本與正樣本的比例')
        self.bcPopularItems = Param(self, 'bcPopularItems', '熱門物品的列表，必須是 Broadcast variable')
        self._setDefault(negativeRatingValue=0, negativePositiveRatio=1, bcPopularItems=None)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, negativeRatingValue=None, negativePositiveRatio=None, bcPopularItems=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setNegativeRatingValue(self, value):
        self._paramMap[self.negativeRatingValue] = value
        return self

    def getNegativeRatingValue(self):
        return self.getOrDefault(self.negativeRatingValue)

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

    def _transform(self, ratingDF):
        negativeRatingValue = self.getNegativeRatingValue()
        negativePositiveRatio = self.getNegativePositiveRatio()
        if negativePositiveRatio <= 0:
            return ratingDF
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
            negativeItemsCount = len(negativeItems)
            for popularItem in popularItems:
                if popularItem not in positiveItems:
                    negativeItems.append(popularItem)
                    negativeItemsCount += 1
                    if negativeItemsCount >= positiveItemsCount * negativePositiveRatio:
                        break
            return (user, negativeItems)

        def expandNegativeSamples(userItemsPair):
            user, negativeItems = userItemsPair
            return ((user, negative, negativeRatingValue) for negative in negativeItems)

        negativeRDD = ratingDF \
            .select('user', 'item') \
            .rdd \
            .aggregateByKey(set(), seqFunc, combFunc) \
            .map(lambda x: getNegativeSamples(x)) \
            .flatMap(lambda x: expandNegativeSamples(x))
        negativeDF = spark.createDataFrame(negativeRDD, ratingDF.schema)

        # avoid "Union can only be performed on tables with the same number of columns" in cross-validation
        columns = ('user', 'item', 'rating')
        balancedDF = ratingDF.select(*columns).union(negativeDF.select(*columns))
        return balancedDF
