Albedo
======

A recommender system for discovering GitHub repos you might like - built with [Apache Spark](https://spark.apache.org/).

**Albedo** is a fictional character in Dan Simmons's [Hyperion Cantos](https://en.wikipedia.org/wiki/Hyperion_Cantos) series. Councilor Albedo is the TechnoCore's AI Advisor to the Hegemony of Man.

## Setup

```bash
$ git clone https://github.com/vinta/albedo.git
$ cd albedo
$ make up
```

## Usage

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

# username: albedo
# password: hyperion
$ make run
$ open http://127.0.0.1:8000/admin/

# you could also create an Apache Spark cluster on Google Cloud Dataproc
# https://cloud.google.com/dataproc/
$ make spark_start

$ make train_als
```

## Related Posts

- [Setup Spark on macOS](https://vinta.ws/code/setup-spark-on-macos.html)
- [Run interactive notebook with Spark and Scala](https://vinta.ws/code/run-interactive-notebook-with-spark-and-scala.html)
- [Build a recommender system with PySpark: Implicit ALS](https://vinta.ws/code/build-a-recommender-system-with-pyspark-implicit-als.html)
