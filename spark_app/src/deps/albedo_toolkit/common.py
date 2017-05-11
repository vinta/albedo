# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType


spark = SparkSession.builder.getOrCreate()

ratingSchema = StructType([
    StructField('user', IntegerType(), nullable=False),
    StructField('item', IntegerType(), nullable=False),
    StructField('rating', IntegerType(), nullable=False),
])


def loadRawData():
    url = 'jdbc:mysql://127.0.0.1:3306/albedo?user=root&password=123&verifyServerCertificate=false&useSSL=false'
    properties = {
        'driver': 'com.mysql.jdbc.Driver',
    }
    rawDF = spark.read.jdbc(url, table='app_repostarring', properties=properties)
    return rawDF
