package ws.vinta.albedo.recommenders

import org.apache.http.HttpHost
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.elasticsearch.action.search.SearchRequest
import org.elasticsearch.client.{RestClient, RestHighLevelClient}
import org.elasticsearch.index.query.MoreLikeThisQueryBuilder.Item
import org.elasticsearch.index.query.QueryBuilders._
import org.elasticsearch.search.SearchHit
import org.elasticsearch.search.builder.SearchSourceBuilder

class ContentRecommender(override val uid: String) extends Recommender {

  def this() = {
    this(Identifiable.randomUID("contentRecommender"))
  }

  override def source = "content"

  override def recommendForUsers(userDF: Dataset[_]): DataFrame = {
    transformSchema(userDF.schema)

    val lowClient = RestClient.builder(new HttpHost("elasticsearch", 9200, "http")).build()
    val highClient = new RestHighLevelClient(lowClient)

    userDF
      .as[(Int, Seq[Int])]
      .flatMap {
        case (userId, itemIds) => {
          val fields = Array("description", "full_name", "language", "topics")
          val texts = Array("")
          val items = itemIds.map((itemId: Int) => new Item("repo", "repo_info_doc", itemId.toString)).toArray
          val queryBuilder = moreLikeThisQuery(fields, texts, items).minTermFreq(1).maxQueryTerms(12)

          val searchSourceBuilder = new SearchSourceBuilder()
          searchSourceBuilder.query(queryBuilder)
          searchSourceBuilder.from(0)
          searchSourceBuilder.size(5)

          val searchRequest = new SearchRequest()
          searchRequest.indices("repo")
          searchRequest.types("repo_info_doc")
          searchRequest.source(searchSourceBuilder)

          val searchResponse = highClient.search(searchRequest)
          val hits = searchResponse.getHits
          val searchHits = hits.getHits

          searchHits.flatMap((searchHit: SearchHit) => {
            val itemId = searchHit.getId
            Array((userId, itemId.toInt, searchHit.getScore))
          })
        }
      }
      .toDF($(userCol), $(itemCol), $(scoreCol))
      .withColumn($(sourceCol), lit(source))

    lowClient.close()

    userDF.toDF()
  }
}