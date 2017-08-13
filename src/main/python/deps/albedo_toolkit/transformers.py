# coding: utf-8

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder.getOrCreate()


class RatingBuilder(Transformer):

    def _transform(self, raw_df):
        rating_df = raw_df \
            .selectExpr('from_user_id AS user', 'repo_id AS item', '1 AS rating', 'starred_at') \
            .orderBy('user', F.col('starred_at').desc())
        return rating_df


# TODO: 不適用新版的資料庫 schema 了，待處理
class DataCleaner(Transformer):

    @keyword_only
    def __init__(self, min_item_stargazers_count=None, max_item_stargazers_count=None, min_user_starred_count=None, max_user_starred_count=None):
        super(DataCleaner, self).__init__()
        self.min_item_stargazers_count = Param(self, 'min_item_stargazers_count', '移除 stargazer 數低於這個數字的 item')
        self.max_item_stargazers_count = Param(self, 'max_item_stargazers_count', '移除 stargazer 數超過這個數字的 item')
        self.min_user_starred_count = Param(self, 'min_user_starred_count', '移除 starred repo 數低於這個數字的 user')
        self.max_user_starred_count = Param(self, 'max_user_starred_count', '移除 starred repo 數超過這個數字的 user')
        self._setDefault(min_item_stargazers_count=1, max_item_stargazers_count=50000, min_user_starred_count=1, max_user_starred_count=50000)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, min_item_stargazers_count=None, max_item_stargazers_count=None, min_user_starred_count=None, max_user_starred_count=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def set_min_item_stargazers_count(self, value):
        self._paramMap[self.min_item_stargazers_count] = value
        return self

    def get_min_item_stargazers_count(self):
        return self.getOrDefault(self.min_item_stargazers_count)

    def set_max_item_stargazers_count(self, value):
        self._paramMap[self.max_item_stargazers_count] = value
        return self

    def get_max_item_stargazers_count(self):
        return self.getOrDefault(self.max_item_stargazers_count)

    def set_min_user_starred_count(self, value):
        self._paramMap[self.min_user_starred_count] = value
        return self

    def get_min_user_starred_count(self):
        return self.getOrDefault(self.min_user_starred_count)

    def set_max_user_starred_count(self, value):
        self._paramMap[self.max_user_starred_count] = value
        return self

    def get_max_user_starred_count(self):
        return self.getOrDefault(self.max_user_starred_count)

    def _transform(self, rating_df):
        min_item_stargazers_count = self.get_min_item_stargazers_count()
        max_item_stargazers_count = self.get_max_item_stargazers_count()
        min_user_starred_count = self.get_min_user_starred_count()
        max_user_starred_count = self.get_max_user_starred_count()

        to_keep_items_df = rating_df \
            .groupBy('item') \
            .agg(F.count('user').alias('stargazers_count')) \
            .where('stargazers_count >= {0} AND stargazers_count <= {1}'.format(min_item_stargazers_count, max_item_stargazers_count)) \
            .orderBy('stargazers_count', ascending=False) \
            .select('item', 'stargazers_count')
        temp1_df = rating_df.join(to_keep_items_df, 'item', 'inner')

        to_keep_users_df = temp1_df \
            .groupBy('user') \
            .agg(F.count('item').alias('starred_count')) \
            .where('starred_count >= {0} AND starred_count <= {1}'.format(min_user_starred_count, max_user_starred_count)) \
            .orderBy('starred_count', ascending=False) \
            .select('user', 'starred_count')
        temp2_df = temp1_df.join(to_keep_users_df, 'user', 'inner')

        clean_df = temp2_df.select('user', 'item', 'rating', 'starred_at')
        return clean_df


class PredictionProcessor(Transformer):

    def _transform(self, predicted_df):
        non_null_df = predicted_df.dropna(subset=['prediction', ])
        prediction_df = non_null_df.withColumn('prediction', non_null_df['prediction'].cast('double'))
        return prediction_df
