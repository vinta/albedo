package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._
import ws.vinta.albedo.closures.UDFs._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object Word2VecRecommender {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("Word2VecRecommender")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val userProfileDF = loadUserProfileDF()

    val repoProfileDF = loadRepoProfileDF()

    val rawStarringDS = loadRawStarringDS()

    //val starringRepoInfoDF = rawStarringDS.join(rawRepoInfoDS, Seq("repo_id"))

    // Prepare Data

    val userTextDF = userProfileDF
      .where($"bio" =!= "")
      .withColumn("text", concat_ws(" ", $"login", $"bio", $"company", $"location"))

    val repoTextDF = repoProfileDF
      .where($"description" =!= "")
      .withColumn("text", concat_ws(" ", $"owner_username", $"name", $"language", $"description", $"topics"))
      // TODO: remove
      .where($"fork" === false)
      .where(!$"description".like("%moved to%"))
      .select($"repo_id", $"text", $"stargazers_count")

    val corpusDF = userTextDF.select($"text").union(repoTextDF.select($"text"))
    corpusDF.cache()

    //val userTopicsDF = starringRepoInfoDF
    //  .where($"topics" =!= "")
    //  .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
    //  .where($"rank" <= 50)
    //  .groupBy($"user_id")
    //  .agg(concat_ws(",", collect_list($"topics")).alias("topics_concat"))
    //  .select($"user_id", $"topics_concat".alias("text"))

    // Split Data

    // 雖然不是每個演算法都需要劃分 training set 和 test set
    // 不過為了方便比較，我們還是統一使用 20% 的 test set 來評估每個模型
    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))
    testDF.cache()

    val testUserDF = testDF.select($"user_id").distinct()

    // Build the Model Pipeline

    val hanLPTokenizer = new HanLPTokenizer()
      .setInputCol("text")
      .setOutputCol("text_words")

    val word2Vec = new Word2Vec()
      .setInputCol("text_words")
      .setOutputCol("text_w2v")
      .setMaxIter(10)
      .setVectorSize(100)
      .setWindowSize(5)
      .setMinCount(10)

    val pipeline = new Pipeline()
      .setStages(Array(hanLPTokenizer, word2Vec))

    // Train the Model

    val pipelineModelPath = s"${settings.dataDir}/${settings.today}/word2VecPipelineModel.parquet"
    val pipelineModel = loadOrCreateModel[PipelineModel](PipelineModel, pipelineModelPath, () => {
      pipeline.fit(corpusDF)
    })

    // Make Recommendations

    val repoVectorDF = pipelineModel.transform(repoTextDF.where($"stargazers_count".between(100, 1500)))

    val repoWordRDD = repoVectorDF
      .select($"repo_id", $"text_w2v")
      .limit(100)
      .rdd
      .flatMap((row: Row) => {
        val repoId = row.getInt(0)
        val vector = row.getAs[DenseVector](1)
        vector.toArray.zipWithIndex.map({
          case (element, index) => MatrixEntry(repoId, index, element)
        })
      })

    val repoWordMatrix = new CoordinateMatrix(repoWordRDD)

    val wordRepoMatrix = repoWordMatrix.transpose

    val repoSimilarityRDD = wordRepoMatrix
      .toRowMatrix
      .columnSimilarities(0.1)
      .entries.map({
        case MatrixEntry(row: Long, col: Long, sim: Double) => (row, col, sim)
      })

    val repoSimilarityDF = spark.createDataFrame(repoSimilarityRDD)
      .toDF("item_1", "item_2", "similarity")
      .where($"similarity" > 0)
      .orderBy($"similarity".desc)

    repoSimilarityDF.show(false)

    //val topK = 30
    //
    ////val lsh = new BucketedRandomProjectionLSH()
    ////  .setBucketLength(4.0)
    ////  .setNumHashTables(5)
    ////  .setInputCol("text_w2v")
    ////  .setOutputCol("text_hashes")
    //
    //val lsh = new MinHashLSH()
    //  .setNumHashTables(5)
    //  .setInputCol("text_w2v")
    //  .setOutputCol("text_hashes")
    //
    //val lshModel = lsh.fit(repoTextVectorDF)
    //
    //val userHashedDF = lshModel.transform(userTextVectorDF)
    //userHashedDF.cache()
    //
    //val repoHashedDF = lshModel.transform(repoTextVectorDF)
    //repoHashedDF.cache()
    //
    //val userSimilarRepoDF = lshModel
    //  .approxSimilarityJoin(userHashedDF, repoHashedDF, 0.005, "distance")
    //  .select($"datasetA.user_id", $"datasetB.repo_id", $"distance")
    //  //.orderBy($"datasetA.user_id".asc, $"distance".asc)
    //
    //println("count: ")
    //userSimilarRepoDF.show(false)
    //
    ////println(s"count: ${userSimilarRepoDF.rdd.countApprox(1000 * 60 * 10, 0.6)}")
    //
    //// TODO
    //import scala.collection.mutable
    //val dfs = mutable.ArrayBuffer.empty[DataFrame]
    //val similarDF = dfs.reduce(_ union _)
    //
    //// Evaluate the Model
    //
    //val userActualItemsDS = testDF
    //  .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))
    //  .as[UserItems]
    //
    //val userPredictedItemsDS = userSimilarRepoDF
    //  .join(testUserDF, Seq("user_id"))
    //  .transform(intoUserPredictedItems($"user_id", $"repo_id", $"distance".asc, topK))
    //  .as[UserItems]
    //
    //val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
    //  .setMetricName("ndcg@k")
    //  .setK(topK)
    //  .setUserCol("user_id")
    //  .setItemsCol("items")
    //val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    //println(s"${rankingEvaluator.getMetricName} = $metric")
    //// NDCG@k = ???

    spark.stop()
  }
}