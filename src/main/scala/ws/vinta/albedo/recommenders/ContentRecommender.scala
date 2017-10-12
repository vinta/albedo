package ws.vinta.albedo.recommenders

import org.apache.http.HttpHost
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.elasticsearch.action.search.SearchRequest
import org.elasticsearch.client.{RestClient, RestHighLevelClient}
import org.elasticsearch.index.query.MoreLikeThisQueryBuilder.Item
import org.elasticsearch.index.query.QueryBuilders._
import org.elasticsearch.search.SearchHit
import org.elasticsearch.search.builder.SearchSourceBuilder
import ws.vinta.albedo.closures.DBFunctions._

class ContentRecommender(override val uid: String) extends Recommender {

  def this() = {
    this(Identifiable.randomUID("contentRecommender"))
  }

  val enableEvaluationMode = new Param[Boolean](this, "enableEvaluationMode", "Should be enable for evaluation only")

  def getEnableEvaluationMode: Boolean = $(enableEvaluationMode)

  def setEnableEvaluationMode(value: Boolean): this.type = set(enableEvaluationMode, value)
  setDefault(enableEvaluationMode -> false)

  override def source = "content"

  override def recommendForUsers(userDF: Dataset[_]): DataFrame = {
    transformSchema(userDF.schema)

    import userDF.sparkSession.implicits._

    val userRecommendedItemDF = userDF
      .as[Int]
      .flatMap {
        case (userId) => {
          // 因為 More Like This query 用 document id 查詢時
          // 結果會過濾掉那些做為條件的 document ids
          // 但是這樣在 evaluate 的時候就不太合適了
          // 所以我們改用後 k 個 repo 當作查詢條件
          val limit = $(topK)
          val offset = if ($(enableEvaluationMode)) $(topK) else 0
          val repoIds = selectUserStarredRepos(userId, limit, offset)

          val lowClient = RestClient.builder(new HttpHost("127.0.0.1", 9200, "http")).build()
          val highClient = new RestHighLevelClient(lowClient)

          val fields = Array("description", "full_name", "language", "topics")
          val texts = Array("")
          val items = repoIds.map((itemId: Int) => new Item("repo", "repo_info_doc", itemId.toString))
          val queryBuilder = moreLikeThisQuery(fields, texts, items)
            .minTermFreq(2)
            .maxQueryTerms(50)

          val searchSourceBuilder = new SearchSourceBuilder()
          searchSourceBuilder.query(queryBuilder)
          searchSourceBuilder.size($(topK))
          searchSourceBuilder.from(0)

          val searchRequest = new SearchRequest()
          searchRequest.indices("repo")
          searchRequest.types("repo_info_doc")
          searchRequest.source(searchSourceBuilder)

          val searchResponse = highClient.search(searchRequest)
          val hits = searchResponse.getHits
          val searchHits = hits.getHits

          val userItemScoreTuples = searchHits.map((searchHit: SearchHit) => {
            val itemId = searchHit.getId.toInt
            val score = searchHit.getScore
            (userId, itemId, score)
          })

          lowClient.close()

          userItemScoreTuples
        }
      }
      .toDF($(userCol), $(itemCol), $(scoreCol))
      .withColumn($(sourceCol), lit(source))

    userRecommendedItemDF
  }
}