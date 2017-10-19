package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._

object Word2VecBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("Word2VecBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadRawUserInfoDS()

    val rawRepoInfoDS = loadRawRepoInfoDS()

    // Build the Pipeline

    val userTextDF = rawUserInfoDS
      .withColumn("text", concat_ws(", ", $"login", $"name", $"company", $"location", $"bio"))
      .select($"user_id", $"login", $"text")

    val repoTextDF = rawRepoInfoDS
      .withColumn("text", concat_ws(", ", $"owner_username", $"name", $"language", $"description", $"topics"))
      .select($"repo_id", $"full_name", $"text")

    val corpusDF = userTextDF.select($"text").union(repoTextDF.select($"text"))
    corpusDF.cache()

    val hanLPTokenizer = new HanLPTokenizer()
      .setInputCol("text")
      .setOutputCol("text_words")
      .setShouldRemoveStopWords(true)

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("text_words")
      .setOutputCol("text_filtered_words")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))

    //val snowballStemmer = new SnowballStemmer()
    //  .setInputCol("text_filtered_words")
    //  .setOutputCol("text_stemmed_words")
    //  .setLanguage("english")

    val word2Vec = new Word2Vec()
      .setInputCol("text_filtered_words")
      .setOutputCol("text_w2v")
      .setMaxIter(20)
      .setVectorSize(200)
      .setWindowSize(5)
      .setMinCount(10)

    val pipeline = new Pipeline()
      .setStages(Array(hanLPTokenizer, stopWordsRemover, word2Vec))

    // Train the Model

    val pipelineModelPath = s"${settings.dataDir}/${settings.today}/corpusPipelineModel.parquet"
    val pipelineModel = loadOrCreateModel[PipelineModel](PipelineModel, pipelineModelPath, () => {
      pipeline.fit(corpusDF)
    })

    val word2vecDF = pipelineModel.transform(corpusDF)
    word2vecDF.sample(false, 0.5).show(500, false)

    spark.stop()
  }
}