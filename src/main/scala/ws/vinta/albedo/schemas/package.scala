package ws.vinta.albedo

import java.sql.Timestamp

import org.apache.spark.sql.types._

package object schemas {
  case class UserInfo(
    id: Int,
    login: String,
    account_type: String,
    name: String,
    company: String,
    blog: String,
    location: String,
    email: String,
    bio: String,
    public_repos: Int,
    public_gists: Int,
    followers: Int,
    following: Int,
    created_at: Timestamp,
    updated_at: Timestamp
  )

  case class RepoInfo(
    id: Int,
    owner_id: Int,
    owner_username: String,
    owner_type: String,
    name: String,
    full_name: String,
    description: String,
    language: String,
    created_at: Timestamp,
    updated_at: Timestamp,
    pushed_at: Timestamp,
    homepage: String,
    size: Int,
    stargazers_count: Int,
    forks_count: Int,
    subscribers_count: Int,
    has_issues: Boolean,
    has_projects: Boolean,
    has_downloads: Boolean,
    has_wiki: Boolean,
    has_pages: Boolean,
    open_issues_count: Int,
    topics: String
  )

  case class UserRelation(
    id: Int,
    from_user_id: Int,
    from_username: String,
    to_user_id: Int,
    to_username: String,
    relation: String
  )

  case class RepoStarring(
    from_user_id: Int,
    from_username: String,
    repo_id: Int,
    repo_full_name: String,
    starred_at: Timestamp,
    starring: Int
  )

  case class PopularRepo(
    repo_id: Int,
    stars: Int
  )

  val fullStarringSchema = StructType(
    Array(
      StructField("id", IntegerType, nullable = false),
      StructField("from_user_id", IntegerType, nullable = false),
      StructField("from_username", StringType, nullable = false),
      StructField("repo_owner_id", IntegerType, nullable = false),
      StructField("repo_owner_username", StringType, nullable = false),
      StructField("repo_owner_type", StringType, nullable = false),
      StructField("repo_id", IntegerType, nullable = false),
      StructField("repo_name", StringType, nullable = false),
      StructField("repo_full_name", StringType, nullable = false),
      StructField("repo_url", StringType, nullable = false),
      StructField("repo_language", StringType, nullable = false),
      StructField("repo_description", StringType, nullable = false),
      StructField("repo_created_at", TimestampType, nullable = false),
      StructField("repo_updated_at", TimestampType, nullable = false),
      StructField("starred_at", TimestampType, nullable = false),
      StructField("stargazers_count", IntegerType, nullable = false),
      StructField("forks_count", IntegerType, nullable = false),
      StructField("starring", IntegerType, nullable = false)
    )
  )

  val simpleStarringSchema = StructType(
    Array(
      StructField("from_user_id", IntegerType, nullable = false),
      StructField("repo_id", IntegerType, nullable = false),
      StructField("starring", IntegerType, nullable = false),
      StructField("starred_at", TimestampType, nullable = false)
    )
  )
}