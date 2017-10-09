package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._

object ContentRecommenderBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ContentRecommenderBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val rawRepoInfoDS = loadRawRepoInfoDS()

    val rawStarringDS = loadRawStarringDS()

    //val userProfileDF = loadUserProfileDF()
    //val repoProfileDF = loadRepoProfileDF()

    //val starringRepoInfoDF = rawStarringDS.join(rawRepoInfoDS, Seq("repo_id"))

    // Prepare Data

    val nullableColumnNames = Array("description", "homepage")

    val unmaintainedWords = Array("%unmaintained%", "%no longer maintained%", "%no longer actively maintained%", "%not maintained%", "%not actively maintained%", "%deprecated%", "%moved to%")
    val assignmentWords = Array("%assignment%")

    val repoTextDF = rawRepoInfoDS
      .na.fill("", nullableColumnNames)
      .where($"fork" === false)
      .where($"stargazers_count".between(1000, 3000))
      .where($"description" =!= "")
      .withColumn("clean_description", lower($"description"))
      .withColumn("is_unmaintained", when(unmaintainedWords.map($"clean_description".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_assignment", when(assignmentWords.map($"clean_description".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .where($"is_unmaintained" === 0 and $"is_assignment" === 0)
      .drop($"is_unmaintained")
      .drop($"is_assignment")
      .withColumn("text", concat_ws(" ", $"owner_username", $"name", $"language", $"description", $"topics"))
      .select($"repo_id", $"text", $"stargazers_count")
    repoTextDF.persist()

    // Split Data

    // 雖然不是每個演算法都需要劃分 training set 和 test set
    // 不過為了方便比較，我們還是統一使用 20% 的 test set 來評估每個模型
    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))
    testDF.cache()

    val meDF = spark.createDataFrame(Seq(
      (652070, "vinta")
    )).toDF("user_id", "username")

    val testUserDF = testDF.select($"user_id").union(meDF.select($"user_id")).distinct()

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
      pipeline.fit(repoTextDF)
    })

    // Make Recommendations

    val repoVectorDF = pipelineModel.transform(repoTextDF)

    val repoWordRDD = repoVectorDF
      .select($"repo_id", $"text_w2v")
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
      .columnSimilarities(0.0001)
      .entries
      .flatMap({
        case MatrixEntry(row: Long, col: Long, sim: Double) => {
          if (sim >= 0.5) {
            Array((row, col, sim))
          }
          else {
            None
          }
        }
      })

    val repoSimilarityDFpath = s"${settings.dataDir}/${settings.today}/repoSimilarityDF.parquet"
    val repoSimilarityDF = loadOrCreateDataFrame(repoSimilarityDFpath, () => {
      spark.createDataFrame(repoSimilarityRDD).toDF("item_1", "item_2", "similarity")
    })

    repoSimilarityDF.show(false)

    // Evaluate the Model

    val topK = 30

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