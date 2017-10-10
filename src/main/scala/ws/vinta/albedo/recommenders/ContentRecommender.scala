package ws.vinta.albedo.recommenders

import org.apache.http.HttpHost
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

  override def source = "content"

  override def recommendForUsers(userDF: Dataset[_]): DataFrame = {
    transformSchema(userDF.schema)

    import userDF.sparkSession.implicits._

    val userRecommendedItemDF = userDF
      .as[Int]
      .flatMap {
        case (userId) => {
          val itemIds = selectUserStarredRepos(userId)

          val lowClient = RestClient.builder(new HttpHost("127.0.0.1", 9200, "http")).build()
          val highClient = new RestHighLevelClient(lowClient)

          val fields = Array("description", "full_name", "language", "topics")
          val texts = Array("")
          val items = itemIds.map((itemId: Int) => new Item("repo", "repo_info_doc", itemId.toString))
          val queryBuilder = moreLikeThisQuery(fields, texts, items)
            .minTermFreq(1)
            .maxQueryTerms(20)

          val searchSourceBuilder = new SearchSourceBuilder()
          searchSourceBuilder.query(queryBuilder)
          searchSourceBuilder.from(0)
          searchSourceBuilder.size($(topK))

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