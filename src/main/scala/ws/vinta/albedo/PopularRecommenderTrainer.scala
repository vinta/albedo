import org.apache.spark.sql.SparkSession

object PopularRecommenderTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession
      .builder()
      .appName("ALSRecommender")
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    spark.stop()
  }
}