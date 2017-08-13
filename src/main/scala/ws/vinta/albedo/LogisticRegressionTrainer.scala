package ws.vinta.albedo

import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions.lit
import ws.vinta.albedo.preprocessors.{NegativeGenerator, popularReposBuilder}
import ws.vinta.albedo.schemas.{PopularRepo, RawStarring}
import ws.vinta.albedo.utils.CommonUtils

import scala.collection.mutable

object LogisticRegressionTrainer {
  val appName = "LogisticRegressionTrainer"

  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    implicit val spark = SparkSession
      .builder()
      .appName(appName)
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._

    val dateFormatter = new java.text.SimpleDateFormat("yyyyMMdd")
    val today = dateFormatter.format(new java.util.Date())

    // load data

    val rawDFpath = s"spark-data/$today/rawDF.parquet"
    val rawDF: Dataset[RawStarring] = try {
      spark.read.parquet(rawDFpath).as[RawStarring]
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val tempRawDF = CommonUtils.loadRawData().withColumn("starring", lit(1))
          tempRawDF.write.parquet(rawDFpath)
          tempRawDF.as[RawStarring]
        } else {
          throw e
        }
      }
    }
    rawDF.cache()

    //val popularRepos: mutable.LinkedHashSet[Int] = popularReposBuilder.transform(rawDF).as[PopularRepo]
    //  .select("item")
    //  .map(row => row(0).asInstanceOf[Int])
    //  .collect()
    //  .to[mutable.LinkedHashSet]
    //
    //val bcPopularRepos = sc.broadcast(popularRepos)
    //val negativeGenerator = new NegativeGenerator(bcPopularRepos)
    //negativeGenerator
    //  .setNegativeValue(0)
    //  .setNegativePositiveRatio(1.0)
    //val balancedDF: DataFrame = negativeGenerator.transform(rawDF)
    //balancedDF.show()
    //println(balancedDF.count())

    spark.stop()
  }
}