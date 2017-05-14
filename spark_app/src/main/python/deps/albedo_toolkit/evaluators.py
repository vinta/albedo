# coding: utf-8

from pyspark import keyword_only
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import Param
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from albedo_toolkit.transformers import PerUserActualItemsBuilder
from albedo_toolkit.transformers import PerUserPredictedItemsBuilder


spark = SparkSession.builder.getOrCreate()


class RankingEvaluator(Evaluator):

    @keyword_only
    def __init__(self, k=30, rawDF=None):
        super(RankingEvaluator, self).__init__()
        self.k = Param(self, 'k', None)
        self.rawDF = Param(self, 'rawDF', None)
        self._setDefault(k=30, rawDF=None)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

        self.perUserActualItemsDF = None

    @keyword_only
    def setParams(self, k=30, rawDF=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setK(self, value):
        self._paramMap[self.k] = value
        return self

    def getK(self):
        return self.getOrDefault(self.k)

    def setRawDF(self, value):
        self._paramMap[self.rawDF] = value
        return self

    def getRawDF(self):
        return self.getOrDefault(self.rawDF)

    def _evaluate(self, outputDF):
        k = self.getK()

        if not self.perUserActualItemsDF:
            rawDF = self.getRawDF()
            perUserActualItemsBuilder = PerUserActualItemsBuilder(k=k)
            perUserActualItemsDF = perUserActualItemsBuilder.transform(rawDF)
            self.perUserActualItemsDF = perUserActualItemsDF
        else:
            perUserActualItemsDF = self.perUserActualItemsDF
        perUserActualItemsDF.cache()

        perUserPredictedItemsBuilder = PerUserPredictedItemsBuilder(k=k)
        perUserPredictedItemsDF = perUserPredictedItemsBuilder.transform(outputDF)

        perUserItemsRDD = perUserPredictedItemsDF.join(F.broadcast(perUserActualItemsDF), 'user', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))
        rankingMetrics = RankingMetrics(perUserItemsRDD)
        metric = rankingMetrics.ndcgAt(k)
        return metric
