package ws.vinta.albedo.utils

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

trait SparkJob {
  val conf = new SparkConf()

  implicit val spark: SparkSession = SparkSession
    .builder()
    .config(conf)
    .master("local[*]")
    .getOrCreate()

  implicit val sc = spark.sparkContext
}