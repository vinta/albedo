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

    @keyword_only
    def __init__(self, minStargazersCount=0):
        super(RatingBuilder, self).__init__()
        self.minStargazersCount = Param(self, 'minStargazersCount', None)
        self._setDefault(minStargazersCount=0)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, minStargazersCount=0):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setMinStargazersCount(self, value):
        self._paramMap[self.minStargazersCount] = value
        return self

    def getMinStargazersCount(self):
        return self.getOrDefault(self.minStargazersCount)

    def _transform(self, rawDF):
        minStargazersCount = self.getMinStargazersCount()
        ratingDF = rawDF \
            .selectExpr('from_user_id AS user', 'repo_id AS item', '1 AS rating') \
            .where('stargazers_count > {0}'.format(minStargazersCount)) \
            .orderBy('user', 'starred_at')
        return ratingDF


class PopularItemsBuilder(Transformer):

    @keyword_only
    def __init__(self, minStargazersCount=0):
        super(PopularItemsBuilder, self).__init__()
        self.minStargazersCount = Param(self, 'minStargazersCount', None)
        self._setDefault(minStargazersCount=0)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, minStargazersCount=0):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setMinStargazersCount(self, value):
        self._paramMap[self.minStargazersCount] = value
        return self

    def getMinStargazersCount(self):
        return self.getOrDefault(self.minStargazersCount)

    def _transform(self, rawDF):
        popularItemsDF = rawDF \
            .where('stargazers_count > 1000') \
            .groupBy('repo_id') \
            .agg(F.max('stargazers_count').alias('stars')) \
            .orderBy('stars', ascending=False) \
            .withColumnRenamed('repo_id', 'item')
        return popularItemsDF


# TODO
class OutlierRemover(Transformer):

    def _transform(self, df):
        cleanDF = df
        return cleanDF


class NegativeGenerator(Transformer):

    @keyword_only
    def __init__(self, negativePositiveRatio=1, bcPopularItems=None):
        super(NegativeGenerator, self).__init__()
        self.negativePositiveRatio = Param(self, 'negativePositiveRatio', None)
        self.bcPopularItems = Param(self, 'bcPopularItems', None)
        self._setDefault(negativePositiveRatio=1, bcPopularItems=None)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, negativePositiveRatio=1, bcPopularItems=None):
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

    def _transform(self, ratingDF):
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


class OutputProcessor(Transformer):

    def _transform(self, predictedDF):
        nonNullDF = predictedDF.dropna(subset=['prediction', ])
        outputDF = nonNullDF.withColumn('prediction', nonNullDF['prediction'].cast('double'))
        return outputDF


class PerUserPredictedItemsBuilder(Transformer):

    @keyword_only
    def __init__(self, k=30):
        super(PerUserPredictedItemsBuilder, self).__init__()
        self.k = Param(self, 'k', None)
        self._setDefault(k=0)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, k=30):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setK(self, value):
        self._paramMap[self.k] = value
        return self

    def getK(self):
        return self.getOrDefault(self.k)

    def _transform(self, outputDF):
        k = self.getK()
        windowSpec = Window.partitionBy('user').orderBy(col('prediction').desc())
        predictedPerUserItemsDF = outputDF \
            .select('user', 'item', 'prediction', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(k)) \
            .groupBy('user') \
            .agg(expr('collect_list(item) as items'))
        return predictedPerUserItemsDF


class PerUserActualItemsBuilder(Transformer):

    @keyword_only
    def __init__(self, k=30):
        super(PerUserActualItemsBuilder, self).__init__()
        self.k = Param(self, 'k', None)
        self._setDefault(k=0)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, k=30):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setK(self, value):
        self._paramMap[self.k] = value
        return self

    def getK(self):
        return self.getOrDefault(self.k)

    def _transform(self, rawDF):
        k = self.getK()
        windowSpec = Window.partitionBy('from_user_id').orderBy(col('starred_at').desc())
        actualPerUserItemsDF = rawDF \
            .select('from_user_id', 'repo_id', 'starred_at', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(k)) \
            .groupBy('from_user_id') \
            .agg(expr('collect_list(repo_id) as items')) \
            .withColumnRenamed('from_user_id', 'user')
        return actualPerUserItemsDF
