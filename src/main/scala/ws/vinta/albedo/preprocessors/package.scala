package ws.vinta.albedo

import org.apache.spark.ml.feature.SQLTransformer

package object preprocessors {
  val simpleStarringBuilder = new SQLTransformer()
  val simpleStarringSQL = """
  SELECT from_user_id, repo_id, 1 AS starring, starred_at
  FROM __THIS__
  ORDER BY from_user_id, starred_at DESC
  """
  simpleStarringBuilder.setStatement(simpleStarringSQL)

  val popularReposBuilder = new SQLTransformer()
  val popularReposSQL = """
  SELECT repo_id, MAX(stargazers_count) AS stars
  FROM __THIS__
  WHERE stargazers_count > 1000
  GROUP BY repo_id
  ORDER BY stars DESC
  """
  popularReposBuilder.setStatement(popularReposSQL)
}