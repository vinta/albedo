package ws.vinta.albedo

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.sql.SparkSession

package object settings {
  private val spark = SparkSession.builder().getOrCreate()
  private val sc = spark.sparkContext

  val dataDir: String = sc.getConf.get("spark.albedo.dataDir", "./spark-data")

  def today: String = {
    val now = LocalDateTime.now()
    val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
    now.format(formatter)
    // 20171026
  }
}