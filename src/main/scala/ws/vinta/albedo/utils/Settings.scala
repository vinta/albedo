package ws.vinta.albedo.utils

import org.apache.spark.sql.SparkSession

object Settings {
  val spark = SparkSession.builder().getOrCreate()
  val sc = spark.sparkContext

  private[albedo] val DATA_DIR = sc.getConf.get("spark.albedo.data_dir")
}
