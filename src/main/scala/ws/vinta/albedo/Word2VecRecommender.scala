package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, Word2Vec}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._
import org.apache.spark.ml.linalg.Vector

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

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRawRepoInfoDS()
    rawRepoInfoDS.cache()

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    val starringRepoInfoDF = rawStarringDS.join(rawRepoInfoDS, Seq("repo_id"))

    // Prepare Data

    val userTextDF = rawUserInfoDS.select($"user_id", concat_ws(" ", $"login", $"bio", $"company", $"location").alias("text"))
    val repoTextDF = rawRepoInfoDS.select($"repo_id", concat_ws(" ", $"owner_username", $"name", $"language", $"description", $"topics").alias("text"))
    val corpusDF = userTextDF.select($"text").union(repoTextDF.select($"text"))
    corpusDF.cache()

    val userTopicsDF = starringRepoInfoDF
      .where($"topics" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(",", collect_list($"topics")).alias("topics_concat"))
      .select($"user_id", $"topics_concat".alias("text"))

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
      .setVectorSize(200)
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

    val topK = 30

    val userTopicsVectorDF = pipelineModel.transform(userTopicsDF)

    val repoTextVectorDF = pipelineModel.transform(repoTextDF)

    val brpLSH = new BucketedRandomProjectionLSH()
      .setBucketLength(8.0)
      .setNumHashTables(50)
      .setInputCol("text_w2v")
      .setOutputCol("text_hashes")
    val brpLSHmodel = brpLSH.fit(repoTextVectorDF)

    //userTopicsVectorDF.flatMap((row: Row) => {
    //  val userID = row(0).asInstanceOf[Int]
    //  val vector = row(1).asInstanceOf[Vector]
    //  val similarDF = brpLSHmodel
    //    .approxNearestNeighbors(repoTextVectorDF, vector, topK)
    //    .orderBy($"distCol".asc)
    //  similarDF.select($"repo_id", $"distCol").collect().map(row => (userID, row(0), row(1)))
    //})

    val userSimilarRepoDF = brpLSHmodel
      .approxSimilarityJoin(userTopicsVectorDF, repoTextVectorDF, 0.05, "distance")
      .select($"datasetA.user_id", $"datasetB.repo_id", $"distance")
      //.orderBy($"datasetA.user_id".asc, $"distance".asc)

    println(s"count: ${userSimilarRepoDF.rdd.countApprox(10000)}")

    // Evaluate the Model

    val userActualItemsDS = testDF
      .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))
      .as[UserItems]

    val userPredictedItemsDS = userSimilarRepoDF
      .join(testUserDF, Seq("user_id"))
      .transform(intoUserPredictedItems($"user_id", $"repo_id", $"distance".asc, topK))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("ndcg@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getMetricName} = $metric")
    // NDCG@k = ???

    spark.stop()
  }
}