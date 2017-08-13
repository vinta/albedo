# coding: utf-8

from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession

from albedo_toolkit.common import load_raw_data
from albedo_toolkit.common import print_cross_validation_parameters
from albedo_toolkit.evaluators import RankingEvaluator
# from albedo_toolkit.transformers import DataCleaner
from albedo_toolkit.transformers import PredictionProcessor
from albedo_toolkit.transformers import RatingBuilder


conf = SparkConf()

spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()

sc = spark.sparkContext

# load data

raw_df = load_raw_data()

# format data

rating_builder = RatingBuilder()
rating_df = rating_builder.transform(raw_df)
rating_df.cache()

# cross-validate models

# data_cleaner = DataCleaner()

als = ALS(implicitPrefs=True, seed=42)

prediction_processor = PredictionProcessor()

pipeline = Pipeline(stages=[
    # data_cleaner,
    als,
    prediction_processor,
])

# .addGrid(data_cleaner.min_item_stargazers_count, [1, 10, 100]) \
# .addGrid(data_cleaner.max_item_stargazers_count, [4000, ]) \
# .addGrid(data_cleaner.min_user_starred_count, [1, 10, 100]) \
# .addGrid(data_cleaner.max_user_starred_count, [1000, 4000, ]) \
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [50, 100]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(als.alpha, [0.01, 0.89, 1, 40, ]) \
    .addGrid(als.maxIter, [22, ]) \
    .build()

ranking_evaluator = RankingEvaluator(k=30)

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=param_grid,
                    evaluator=ranking_evaluator,
                    numFolds=2)

cv_model = cv.fit(rating_df)

# show results

print_cross_validation_parameters(cv_model)
