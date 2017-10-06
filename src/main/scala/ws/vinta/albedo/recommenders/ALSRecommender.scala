package ws.vinta.albedo.recommenders

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import ws.vinta.albedo.settings

class ALSRecommender(override val uid: String) extends Recommender {

  def this() = {
    this(Identifiable.randomUID("alsRecommender"))
  }

  private def alsModel: ALSModel = {
    val alsModelPath = s"${settings.dataDir}/${settings.today}/alsModel.parquet"
    ALSModel.load(alsModelPath)
  }

  def blockify(factors: Dataset[(Int, Array[Float])], blockSize: Int = 4096): Dataset[Seq[(Int, Array[Float])]] = {
    import factors.sparkSession.implicits._
    factors.mapPartitions(_.grouped(blockSize))
  }

  override def source = "als"

  override def recommendForUsers(userDF: Dataset[_]): DataFrame = {
    transformSchema(userDF.schema)

    import userDF.sparkSession.implicits._

    val activeUsers = userDF.select(col($(userCol)).alias("id"))
    val userFactors = alsModel.userFactors.join(activeUsers, Seq("id"))
    val itemFactors = alsModel.itemFactors
    val rank = alsModel.rank
    val num = $(topK)

    val userFactorsBlocked = blockify(userFactors.as[(Int, Array[Float])])
    val itemFactorsBlocked = blockify(itemFactors.as[(Int, Array[Float])])
    val ratings = userFactorsBlocked.crossJoin(itemFactorsBlocked)
      .as[(Seq[(Int, Array[Float])], Seq[(Int, Array[Float])])]
      .flatMap { case (srcIter, dstIter) =>
        val m = srcIter.size
        val n = math.min(dstIter.size, num)
        val output = new Array[(Int, Int, Float)](m * n)
        var i = 0
        val pq = new BoundedPriorityQueue[(Int, Float)](num)(Ordering.by(_._2))
        srcIter.foreach { case (srcId, srcFactor) =>
          dstIter.foreach { case (dstId, dstFactor) =>
            val score = new F2jBLAS().sdot(rank, srcFactor, 1, dstFactor, 1)
            pq += dstId -> score
          }
          pq.foreach { case (dstId, score) =>
            output(i) = (srcId, dstId, score)
            i += 1
          }
          pq.clear()
        }
        output.toSeq
      }

    //ratings.cache()
    //
    //val topKAggregator = new TopByKeyAggregator[Int, Int, Float](num, Ordering.by(_._2))
    //val recs = ratings.as[(Int, Int, Float)]
    //  .groupByKey(_._1)
    //  .agg(topKAggregator.toColumn)
    //  .toDF("id", "recommendations")
    //val arrayType = ArrayType(
    //  new StructType()
    //    .add($(itemCol), IntegerType)
    //    .add("rating", FloatType)
    //)
    //val df = recs.select($"id".as($(userCol)), $"recommendations".cast(arrayType))
    //df.show(false)

    ratings
      .toDF($(userCol), $(itemCol), $(scoreCol))
      .withColumn($(sourceCol), lit(source))
  }
}