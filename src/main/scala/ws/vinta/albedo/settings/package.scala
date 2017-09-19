package ws.vinta.albedo

import org.apache.spark.sql.SparkSession

package object settings {
  private val spark = SparkSession.builder().getOrCreate()
  private val sc = spark.sparkContext

  val dataDir: String = sc.getConf.get("spark.albedo.dataDir", "./spark-data")
  //val dataDir: String = "/mnt/albedo_s3/spark-data"

  def today: String = {
    val dateFormatter = new java.text.SimpleDateFormat("yyyyMMdd")
    dateFormatter.format(new java.util.Date())
  }

  val emptyStringPlaceholder = "__empty"
}