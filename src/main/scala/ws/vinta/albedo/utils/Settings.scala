package ws.vinta.albedo.utils

import org.apache.spark.sql.SparkSession

object Settings {
  private val spark = SparkSession.builder().getOrCreate()
  private val sc = spark.sparkContext

  val dataDir: String = sc.getConf.get("spark.albedo.dataDir", "./spark-data")

  def today: String = {
    val dateFormatter = new java.text.SimpleDateFormat("yyyyMMdd")
    dateFormatter.format(new java.util.Date())
  }
}