# coding: utf-8

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder.getOrCreate()


def loadRawData():
    url = 'jdbc:mysql://127.0.0.1:3306/albedo?user=root&password=123&verifyServerCertificate=false&useSSL=false'
    properties = {'driver': 'com.mysql.jdbc.Driver'}
    rawDF = spark.read.jdbc(url, table='app_repostarring', properties=properties)
    return rawDF


def calculateSparsity(ratingDF):
    result = ratingDF.agg(F.count('rating'), F.countDistinct('user'), F.countDistinct('item')).collect()[0]
    totalUserCount = result['count(DISTINCT user)']
    totalItemCount = result['count(DISTINCT item)']
    zonZeroRatingCount = result['count(rating)']
    density = (zonZeroRatingCount / (totalUserCount * totalItemCount)) * 100
    sparsity = 100 - density
    return sparsity


def randomSplitByUser(df, weights, seed=None):
    trainingRation = weights[0]
    fractions = {row['user']: trainingRation for row in df.select('user').distinct().collect()}
    training = df.sampleBy('user', fractions, seed)
    testRDD = df.rdd.subtract(training.rdd)
    test = spark.createDataFrame(testRDD, df.schema)
    return training, test


def printCrossValidationParameters(cvModel):
    metric_params_pairs = list(zip(cvModel.avgMetrics, cvModel.getEstimatorParamMaps()))
    metric_params_pairs.sort(key=lambda x: x[0], reverse=True)
    for pair in metric_params_pairs:
        metric, params = pair
        print('metric', metric)
        for k, v in params.items():
            print(k.name, v)
        print('')


def recommendItems(rawDF, alsModel, username, topN=30, excludeKnownItems=False):
    userID = rawDF \
        .where('from_username = "{0}"'.format(username)) \
        .select('from_user_id') \
        .take(1)[0]['from_user_id']

    userItemsDF = alsModel \
        .itemFactors. \
        selectExpr('{0} AS user'.format(userID), 'id AS item')
    if excludeKnownItems:
        userKnownItemsDF = rawDF \
            .where('from_user_id = {0}'.format(userID)) \
            .selectExpr('repo_id AS item')
        userItemsDF = userItemsDF.join(userKnownItemsDF, 'item', 'left_anti')

    userPredictedDF = alsModel \
        .transform(userItemsDF) \
        .select('item', 'prediction') \
        .orderBy('prediction', ascending=False) \
        .limit(topN)

    repoDF = rawDF \
        .groupBy('repo_id', 'repo_full_name', 'repo_language') \
        .agg(F.max('stargazers_count').alias('stargazers_count'))

    recommendedItemsDF = userPredictedDF \
        .join(repoDF, userPredictedDF['item'] == repoDF['repo_id'], 'inner') \
        .select('prediction', 'repo_full_name', 'repo_language', 'stargazers_count') \
        .orderBy('prediction', ascending=False)

    return recommendedItemsDF
