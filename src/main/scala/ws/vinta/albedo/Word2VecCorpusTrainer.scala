package ws.vinta.albedo

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._

object Word2VecCorpusTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("Word2VecCorpusTrainer")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRawRepoInfoDS()
    rawRepoInfoDS.cache()

    // Prepare Data

    val userTextDF = rawUserInfoDS.select(concat_ws(" ", $"login", $"bio", $"company", $"location").alias("text"))
    val repoTextDF = rawRepoInfoDS.select(concat_ws(" ", $"owner_username", $"name", $"description", $"topics", $"language").alias("text"))
    val textDF = userTextDF.union(repoTextDF)
    textDF.cache()

    // Train the Model

    val word2VecModelSavePath = s"${settings.dataDir}/${settings.today}/corpusWord2VecModel.parquet"
    val word2VecModel = try {
      Word2VecModel.load(word2VecModelSavePath)
    } catch {
      case e: InvalidInputException => {
        if (e.getMessage.contains("Input path does not exist")) {
          val hanLPTokenizer = new HanLPTokenizer()
            .setInputCol("text")
            .setOutputCol("words")
          val wordsDF = hanLPTokenizer.transform(textDF)

          val word2Vec = new Word2Vec()
            .setInputCol("words")
            .setOutputCol("words_w2v")
            .setMaxIter(10)
            .setVectorSize(200)
            .setWindowSize(5)
            .setMinCount(10)
          val word2VecModel = word2Vec.fit(wordsDF)

          word2VecModel.write.overwrite().save(word2VecModelSavePath)
          word2VecModel
        } else {
          throw e
        }
      }
    }

    word2VecModel.findSynonyms("spark", 5).show(false)
    word2VecModel.findSynonyms("django", 5).show(false)
    word2VecModel.findSynonyms("中文", 5).show(false)

    spark.stop()
  }
}