# coding: utf-8

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark

import numpy as np


conf = SparkConf()
conf.set("spark.executor.memory", "8G")
conf.set("spark.driver.memory", "4G")
conf.set("spark.executor.cores", "4")
conf.set("spark.default.parallelism", "4")
conf.setMaster('local[4]')

spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .appName("albedo") \
    .getOrCreate()
