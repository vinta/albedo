package ws.vinta.albedo.closures

import java.sql.DriverManager
import java.util.Properties

import scala.collection.mutable.ArrayBuffer

object DBFunctions {
  def selectUserStarredRepos(userId: Int, limit: Int, offset: Int): Array[Int] = {
    val dbUrl = "jdbc:mysql://127.0.0.1:3306/albedo?verifyServerCertificate=false&useSSL=false&rewriteBatchedStatements=true"
    val props = new Properties()
    props.setProperty("driver", "com.mysql.jdbc.Driver")
    props.setProperty("user", "root")
    props.setProperty("password", "123")

    val connection = DriverManager.getConnection(dbUrl, props)
    val statement = connection.createStatement()
    val resultSet = statement.executeQuery(s"""
    SELECT repo_id
    FROM app_repostarring
    WHERE user_id = $userId
    ORDER BY starred_at DESC
    LIMIT $limit
    OFFSET $offset;
    """.stripMargin(' '))

    val repoIds = ArrayBuffer.empty[Int]

    while (resultSet.next()) {
      val repoId = resultSet.getInt("repo_id")
      repoIds += repoId
    }

    connection.close()

    repoIds.toArray
  }
}