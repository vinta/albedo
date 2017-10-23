package ws.vinta.albedo

package object schemas {
  case class UserInfo(
    user_id: Int,
    user_login: String,
    user_account_type: String,
    user_name: String,
    user_company: String,
    user_blog: String,
    user_location: String,
    user_email: String,
    user_bio: String,
    user_public_repos_count: Int,
    user_public_gists_count: Int,
    user_followers_count: Int,
    user_following_count: Int,
    user_created_at: java.sql.Timestamp,
    user_updated_at: java.sql.Timestamp
  )

  case class RepoInfo(
    repo_id: Int,
    repo_owner_id: Int,
    repo_owner_username: String,
    repo_owner_type: String,
    repo_name: String,
    repo_full_name: String,
    repo_description: String,
    repo_language: String,
    repo_created_at: java.sql.Timestamp,
    repo_updated_at: java.sql.Timestamp,
    repo_pushed_at: java.sql.Timestamp,
    repo_homepage: String,
    repo_size: Int,
    repo_stargazers_count: Int,
    repo_forks_count: Int,
    repo_subscribers_count: Int,
    repo_is_fork: Boolean,
    repo_has_issues: Boolean,
    repo_has_projects: Boolean,
    repo_has_downloads: Boolean,
    repo_has_wiki: Boolean,
    repo_has_pages: Boolean,
    repo_open_issues_count: Int,
    repo_topics: String
  )

  case class Starring(
    user_id: Int,
    repo_id: Int,
    starred_at: java.sql.Timestamp,
    starring: Double
  )

  case class Relation(
    from_user_id: Int,
    from_username: String,
    to_user_id: Int,
    to_username: String,
    relation: String
  )

  case class PopularRepo(repo_id: Int, stargazers_count: Int)
  case class UserPopularRepo(user_id: Int, repo_id: Int, stargazers_count: Int)

  case class Recommendation(repo_id: Int, rating: Float)
  case class UserRecommendations(user_id: Int, recommendations: Seq[Recommendation])

  case class UserItems(user_id: Int, items: Seq[Int])
}