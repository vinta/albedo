package ws.vinta.albedo

import com.databricks.spark.avro._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Word2Vec
import ws.vinta.albedo.utils.DataSourceUtils.{dataDir, today}
import ws.vinta.albedo.utils.Settings

object GitHubCorpusTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession
      .builder()
      .appName("GitHubCorpusTrainer")
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._

    println("GitHubCorpusTrainer 123")

    //println(s"---${Settings.DATA_DIR}---")

    //val repoDescriptionDF = spark.read.avro(s"$dataDir/spark-data/githubarchive_repo_info.avro")
    //repoDescriptionDF.cache()

    //println(repoDescriptionDF.count())

    //val regexTokenizer = new RegexTokenizer()
    //  .setToLowercase(true)
    //  .setInputCol("description")
    //  .setOutputCol("words")
    //  .setPattern("[\\w-]+").setGaps(false)
    //val tokenizedDF = regexTokenizer.transform(repoDescriptionDF)
    //
    //val stopWordsRemover = new StopWordsRemover()
    //  .setInputCol("words")
    //  .setOutputCol("filtered_words")
    //val filteredDF = stopWordsRemover.transform(tokenizedDF)
    //
    //val word2Vec = new Word2Vec()
    //  .setInputCol("filtered_words")
    //  .setOutputCol("word2vec")
    //  .setVectorSize(10)
    //  .setWindowSize(5)
    //  .setMinCount(0)
    //val word2VecModel = word2Vec.fit(filteredDF)
    //
    //
    //val savePath = s"$dataDir/spark-data/$today/word2VecModel.parquet"
    //word2VecModel.save(savePath)

    spark.stop()
  }
}