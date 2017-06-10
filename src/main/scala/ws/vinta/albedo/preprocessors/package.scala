package ws.vinta.albedo

import org.apache.spark.ml.feature.SQLTransformer

package object preprocessors {
  val starringBuilder = new SQLTransformer()
  val starringSQL = """
  SELECT from_user_id AS user, repo_id AS item, 1 AS star, starred_at
  FROM __THIS__
  ORDER BY user, starred_at DESC
  """
  starringBuilder.setStatement(starringSQL)

  val popularItemsBuilder = new SQLTransformer()
  val popularItemsSQL = """
  SELECT repo_id AS item, MAX(stargazers_count) AS stars
  FROM __THIS__
  WHERE stargazers_count > 1000
  GROUP BY repo_id
  ORDER BY stars DESC
  """
  popularItemsBuilder.setStatement(popularItemsSQL)
}