package ws.vinta.albedo

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Properties

import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit

object Playground {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_WITH_INTELLIJ", "false") == "true") {
      conf.setMaster("local[*]")
      conf.set("spark.driver.memory", "12g")
      //conf.setMaster("local-cluster[1, 3, 12288]")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
      //conf.setMaster("spark://localhost:7077")
      //conf.set("spark.driver.memory", "2g")
      //conf.set("spark.executor.cores", "3")
      //conf.set("spark.executor.memory", "12g")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("Playground")
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext

    //val df = spark.createDataFrame(Seq(
    //  (1, 1, 1),
    //  (1, 2, 0),
    //  (2, 1, 1)
    //)).toDF("user_id", "repo_id", "starring")
    //df.show()

    //Thread.sleep(1000 * 60 * 60)

    val dbUrl = "jdbc:mysql://127.0.0.1:3306/albedo?verifyServerCertificate=false&useSSL=false&rewriteBatchedStatements=true"
    val props = new Properties()
    props.setProperty("driver", "com.mysql.jdbc.Driver")
    props.setProperty("user", "root")
    props.setProperty("password", "123")

    val rawStarringDS = spark.read.jdbc(dbUrl, "app_repostarring", props)
      .select("user_id", "repo_id", "starred_at")
      .withColumn("starring", lit(1.0))

    val als = new ALS()
      .setImplicitPrefs(true)
      .setRank(50)
      .setRegParam(0.5)
      .setAlpha(40)
      .setMaxIter(26)
      .setSeed(42)
      .setColdStartStrategy("drop")
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setRatingCol("starring")
    val alsModel = als.fit(rawStarringDS)

    val now = LocalDateTime.now()
    val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
    val today = now.format(formatter)

    val alsModelPath = s"spark-data/$today/alsModel.parquet"
    alsModel.write.overwrite().save(alsModelPath)

    val predictionDF = alsModel.transform(rawStarringDS)
    predictionDF.show()

    spark.stop()
  }
}