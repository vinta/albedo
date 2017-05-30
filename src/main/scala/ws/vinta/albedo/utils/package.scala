package ws.vinta.albedo

import java.util.Properties

import org.apache.spark.sql.{DataFrame, SparkSession}

package object utils {
  def loadRawData()(implicit spark: SparkSession): DataFrame = {
    val dbUrl = "jdbc:mysql://127.0.0.1:3306/albedo?user=root&password=123&verifyServerCertificate=false&useSSL=false"
    val props = new Properties()
    props.put("driver", "com.mysql.jdbc.Driver")

    spark.read.jdbc(dbUrl, "app_repostarring", props)
  }
}