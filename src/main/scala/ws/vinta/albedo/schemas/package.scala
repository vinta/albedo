package ws.vinta.albedo

import java.sql.Timestamp

import org.apache.spark.sql.types._

package object schemas {
  case class UserInfo(
    user_id: Int,
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

  case class UserRelation(
    from_user_id: Int,
    from_username: String,
    to_user_id: Int,
    to_username: String,
    relation: String
  )

  case class RepoInfo(
    repo_id: Int,
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

  case class RepoStarring(
    user_id: Int,
    repo_id: Int,
    starred_at: Timestamp,
    starring: Double
  )

  case class Recommendation(repo_id: Int, rating: Float)
  case class UserRecommendations(user_id: Int, recommendations: Seq[Recommendation])

  case class UserItems(user_id: Int, items: Seq[Int])
}