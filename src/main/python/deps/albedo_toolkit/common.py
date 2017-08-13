# coding: utf-8

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder.getOrCreate()


def load_raw_data():
    url = 'jdbc:mysql://127.0.0.1:3306/albedo?user=root&password=123&verifyServerCertificate=false&useSSL=false'
    properties = {'driver': 'com.mysql.jdbc.Driver'}
    raw_df = spark.read.jdbc(url, table='app_repostarring', properties=properties)
    return raw_df


def calculate_sparsity(rating_df):
    result = rating_df.agg(F.count('rating'), F.countDistinct('user'), F.countDistinct('item')).collect()[0]
    total_user_count = result['count(DISTINCT user)']
    total_item_count = result['count(DISTINCT item)']
    zon_zero_rating_count = result['count(rating)']
    density = (zon_zero_rating_count / (total_user_count * total_item_count)) * 100
    sparsity = 100 - density
    return sparsity


def random_split_by_user(df, weights, seed=None):
    training_ration = weights[0]
    fractions = {row['user']: training_ration for row in df.select('user').distinct().collect()}
    training = df.sampleBy('user', fractions, seed)
    test_rdd = df.rdd.subtract(training.rdd)
    test = spark.createDataFrame(test_rdd, df.schema)
    return training, test


def print_cross_validation_parameters(cv_model):
    metric_params_pairs = list(zip(cv_model.avgMetrics, cv_model.getEstimatorParamMaps()))
    metric_params_pairs.sort(key=lambda x: x[0], reverse=True)
    for pair in metric_params_pairs:
        metric, params = pair
        print('metric', metric)
        for k, v in params.items():
            print(k.name, v)
        print('')


def recommend_items(raw_df, als_model, username, top_n=30, exclude_known_items=False):
    user_id = raw_df \
        .where('from_username = "{0}"'.format(username)) \
        .select('from_user_id') \
        .take(1)[0]['from_user_id']

    user_items_df = als_model \
        .itemFactors. \
        selectExpr('{0} AS user'.format(user_id), 'id AS item')
    if exclude_known_items:
        user_known_items_df = raw_df \
            .where('from_user_id = {0}'.format(user_id)) \
            .selectExpr('repo_id AS item')
        user_items_df = user_items_df.join(user_known_items_df, 'item', 'left_anti')
    user_predicted_df = als_model \
        .transform(user_items_df) \
        .select('item', 'prediction') \
        .orderBy('prediction', ascending=False) \
        .limit(top_n)
    recommended_items_df = user_predicted_df \
        .join(raw_df, user_predicted_df['item'] == raw_df['repo_id'], 'inner') \
        .select('prediction', 'repo_full_name') \
        .orderBy('prediction', ascending=False)

    return recommended_items_df
