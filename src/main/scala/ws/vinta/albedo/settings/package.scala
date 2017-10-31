package ws.vinta.albedo

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.sql.SparkSession

package object settings {
  private val spark = SparkSession.builder().getOrCreate()
  private val sc = spark.sparkContext

  val dataDir: String = sc.getConf.get("spark.albedo.dataDir", "./spark-data")
  val checkpointDir: String = sc.getConf.get("spark.albedo.checkpointDir", "./spark-data/checkpoint")

  def today: String = {
    val now = LocalDateTime.now()
    val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
    now.format(formatter)
  }

  def md5(text: String): String = {
    java.security.MessageDigest.getInstance("MD5").digest(text.getBytes()).map(0xFF & _).map { "%02x".format(_) }.foldLeft(""){_ + _}
  }
}