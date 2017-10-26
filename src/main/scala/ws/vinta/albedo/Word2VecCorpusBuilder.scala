package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.transformers._
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._

object Word2VecCorpusBuilder {
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
      .appName("Word2VecCorpusBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadRawUserInfoDS().cache()

    val rawRepoInfoDS = loadRawRepoInfoDS().cache()

    // Train the Model

    val columnName = "text"

    val userTextDF = rawUserInfoDS
      .withColumn(columnName, concat_ws(", ", $"user_login", $"user_name", $"user_bio", $"user_company", $"user_location"))
      .select("user_id", "user_login", columnName)

    val repoTextDF = rawRepoInfoDS
      .withColumn(columnName, concat_ws(", ", $"repo_owner_username", $"repo_name", $"repo_language", $"repo_description", $"repo_topics"))
      .select("repo_id", "repo_full_name", columnName)

    val corpusDF = userTextDF.select(columnName)
      .union(repoTextDF.select(columnName))
      .cache()

    val hanLPTokenizer = new HanLPTokenizer()
      .setInputCol(columnName)
      .setOutputCol(s"${columnName}_words")
      .setShouldRemoveStopWords(true)
    val tokenizedDF = hanLPTokenizer.transform(corpusDF)

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(s"${columnName}_words")
      .setOutputCol(s"${columnName}_filtered_words")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
    val filteredDF = stopWordsRemover.transform(tokenizedDF)

    val finalDF = filteredDF.cache()

    val word2VecModelPath = s"${settings.dataDir}/${settings.today}/word2VecModel.parquet"
    val word2VecModel = loadOrCreateModel[Word2VecModel](Word2VecModel, word2VecModelPath, () => {
      val word2Vec = new Word2Vec()
        .setInputCol(s"${columnName}_filtered_words")
        .setOutputCol(s"${columnName}_w2v")
        .setMaxIter(30)
        .setVectorSize(200)
        .setWindowSize(5)
        .setMinCount(10)
      word2Vec.fit(finalDF)
    })

    val word2VecDF = word2VecModel.transform(finalDF)
    word2VecDF.show(false)

    spark.stop()
  }
}