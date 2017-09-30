Albedo
======

A recommender system for discovering GitHub repos, built with [Apache Spark](https://spark.apache.org/).

**Albedo** is a fictional character in Dan Simmons's [Hyperion Cantos](https://en.wikipedia.org/wiki/Hyperion_Cantos) series. Councilor Albedo is the TechnoCore's AI advisor to the Hegemony of Man.

## Setup

```bash
$ git clone https://github.com/vinta/albedo.git
$ cd albedo
$ make up
```

## Collect Data

You need to create your own `GITHUB_PERSONAL_TOKEN` on [your GitHub settings page](https://help.github.com/articles/creating-an-access-token-for-command-line-use/).

```bash
# get into the main container
$ make attach

# this step might take a few hours to complete
# depends on how many repos you starred and how many users you followed
$ (container) python manage.py migrate
$ (container) python manage.py collect_data -t GITHUB_PERSONAL_TOKEN -u GITHUB_USERNAME
# or
$ (container) wget https://s3-ap-northeast-1.amazonaws.com/files.albedo.one/albedo.sql
$ (container) mysql -h mysql -u root -p123 albedo < albedo.sql

# username: albedo
# password: hyperion
$ make run
$ open http://127.0.0.1:8000/admin/
```

## Start a Spark Cluster

You could also create a Spark cluster on [Google Cloud Dataproc](https://cloud.google.com/dataproc/).

```bash
$ make spark_start
```

## Use Popularity as the Recommendation Baseline

See [PopularityRecommenderTrainer.scala](src/main/scala/ws/vinta/albedo/PopularityRecommender.scala) for complete code.

```bash
$ spark-submit \
    --driver-memory 4g \
    --executor-cores 4 \
    --executor-memory 12g \
    --master spark://localhost:7077 \
    --packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
    --class ws.vinta.albedo.PopularityRecommenderTrainer \
    target/albedo-1.0.0-SNAPSHOT.jar
# NDCG@k = 0.0013035714524256231
```

## Build the User Profile for Feature Engineering

See [UserProfileBuilder.scala](src/main/scala/ws/vinta/albedo/UserProfileBuilder.scala) for complete code.

```bash
$ spark-submit \
    --driver-memory 4g \
    --executor-cores 4 \
    --executor-memory 12g \
    --master spark://localhost:7077 \
    --packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
    --class ws.vinta.albedo.UserProfileBuilder \
    target/albedo-1.0.0-SNAPSHOT.jar
```

## Build the Item Profile for Feature Engineering

See [RepoProfileBuilder.scala](src/main/scala/ws/vinta/albedo/RepoProfileBuilder.scala) for complete code.

```bash
$ spark-submit \
    --driver-memory 4g \
    --executor-cores 4 \
    --executor-memory 12g \
    --master spark://localhost:7077 \
    --packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
    --class ws.vinta.albedo.RepoProfileBuilder \
    target/albedo-1.0.0-SNAPSHOT.jar
```

## Train an ALS Model for Candidate Generation

See [ALSRecommender.scala](src/main/scala/ws/vinta/albedo/ALSRecommender.scala) and [ALSRecommenderCV.scala](src/main/scala/ws/vinta/albedo/ALSRecommenderCV.scala) for complete code.

```bash
$ spark-submit \
    --driver-memory 4g \
    --executor-cores 4 \
    --executor-memory 12g \
    --master spark://localhost:7077 \
    --packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
    --class ws.vinta.albedo.ALSRecommenderCV \
    target/albedo-1.0.0-SNAPSHOT.jar

$ spark-submit \
    --driver-memory 4g \
    --executor-cores 4 \
    --executor-memory 12g \
    --master spark://localhost:7077 \
    --packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
    --class ws.vinta.albedo.ALSRecommender \
    target/albedo-1.0.0-SNAPSHOT.jar
# NDCG@k = 0.05026158143766048
```

## Train a Logistic Regression Model for Ranking

See [LogisticRegressionRanker.scala](src/main/scala/ws/vinta/albedo/LogisticRegressionRanker.scala) for complete code.

```bash
$ spark-submit \
    --driver-memory 4g \
    --executor-cores 4 \
    --executor-memory 12g \
    --master spark://localhost:7077 \
    --packages "com.github.fommil.netlib:all:1.1.2" \
    --class ws.vinta.albedo.LogisticRegressionRanker \
    target/albedo-1.0.0-SNAPSHOT.jar
```

## TODO

- Build a recommender system with Spark: Content-based
- Build a recommender system with Spark: GDBT for Feature Learning
- Build a recommender system with Spark: Factorization Machine
- Build a recommender system with Spark: Item2Vec
- Build a recommender system with Spark: XGBoost

## Related Posts

- [Build a recommender system with PySpark: Implicit ALS](https://vinta.ws/code/build-a-recommender-system-with-pyspark-implicit-als.html)
- [Spark SQL cookbook (Scala)](https://vinta.ws/code/spark-sql-cookbook-scala.html)
- [Spark ML cookbook (Scala)](https://vinta.ws/code/spark-ml-cookbook-scala.html)
- [Feature Engineering 特徵工程中常見的方法](https://vinta.ws/code/feature-engineering.html)
