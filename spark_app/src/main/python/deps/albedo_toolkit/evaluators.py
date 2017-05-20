# coding: utf-8

from pyspark import keyword_only
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import Param
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import expr
import pyspark.sql.functions as F


spark = SparkSession.builder.getOrCreate()


class RankingEvaluator(Evaluator):

    @keyword_only
    def __init__(self, k=None):
        super(RankingEvaluator, self).__init__()
        self.k = Param(self, 'k', 'Top K')
        self._setDefault(k=30)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, k=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def isLargerBetter(self):
        return True

    def setK(self, value):
        self._paramMap[self.k] = value
        return self

    def getK(self):
        return self.getOrDefault(self.k)

    def _evaluate(self, predictedDF):
        k = self.getK()

        windowSpec = Window.partitionBy('user').orderBy(col('prediction').desc())
        perUserPredictedItemsDF = predictedDF \
            .select('user', 'item', 'prediction', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(k)) \
            .groupBy('user') \
            .agg(expr('collect_list(item) as items'))

        windowSpec = Window.partitionBy('user').orderBy(col('starred_at').desc())
        perUserActualItemsDF = predictedDF \
            .select('user', 'item', 'starred_at', F.rank().over(windowSpec).alias('rank')) \
            .where('rank <= {0}'.format(k)) \
            .groupBy('user') \
            .agg(expr('collect_list(item) as items'))

        perUserItemsRDD = perUserPredictedItemsDF.join(F.broadcast(perUserActualItemsDF), 'user', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

        if perUserItemsRDD.isEmpty():
            return 0.0

        rankingMetrics = RankingMetrics(perUserItemsRDD)
        metric = rankingMetrics.ndcgAt(k)
        return metric
