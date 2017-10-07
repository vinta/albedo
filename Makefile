.PHONY: clean
clean:
	find . \( -name \*.pyc -o -name \*.pyo -o -name __pycache__ \) -prune -exec rm -rf {} +

.PHONY: up
up:
	mkdir -p ../albedo-vendors/bin
	mkdir -p ../albedo-vendors/dist-packages
	docker-compose up

.PHONY: stop
stop:
	docker-compose stop

.PHONY: attach
attach:
	docker exec -i -t albedo_django_1 bash

.PHONY: install
install:
	docker exec -i -t albedo_django_1 pip install -r requirements.txt

.PHONY: run
run:
	docker exec -i -t albedo_django_1 python manage.py runserver 0.0.0.0:8000

.PHONY: spark_start
spark_start:
	cd ${SPARK_HOME} && ./sbin/start-master.sh -h 0.0.0.0
	cd ${SPARK_HOME} && ./sbin/start-slave.sh spark://localhost:7077

.PHONY: spark_stop
spark_stop:
	cd ${SPARK_HOME} && ./sbin/stop-master.sh
	cd ${SPARK_HOME} && ./sbin/stop-slave.sh

.PHONY: pyspark_notebook
pyspark_notebook:
	find . -name __pycache__ | xargs rm -Rf
	cd src/main/python/deps/ && zip -r ../deps.zip *
	PYSPARK_DRIVER_PYTHON="jupyter" \
	PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip 0.0.0.0" \
	pyspark \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--driver-memory 4g \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--py-files src/main/python/deps.zip

.PHONY: zeppelin_start
zeppelin_start:
	zeppelin-daemon.sh start
	open http://localhost:8080/
	open http://localhost:4040/jobs/

.PHONY: zeppelin_stop
zeppelin_stop:
	zeppelin-daemon.sh stop

.PHONY: build_jar
build_jar:
	mvn clean package -DskipTests

.PHONY: baseline
baseline:
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.PopularityRecommenderTrainer \
	target/albedo-1.0.0-SNAPSHOT.jar

.PHONY: cv_als
cv_als:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--cluster albedo \
	--properties 'spark.executor.memory=11823m,spark.jars.packages=mysql:mysql-connector-java:5.1.41,spark.albedo.dataDir=gs://albedo/spark-data' \
	--class ws.vinta.albedo.ALSRecommenderCV \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.ALSRecommenderCV \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: train_als
train_als:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--cluster albedo \
	--properties 'spark.executor.memory=13312m,spark.jars.packages=mysql:mysql-connector-java:5.1.41,spark.albedo.dataDir=gs://albedo/spark-data' \
	--class ws.vinta.albedo.ALSRecommender \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.ALSRecommender \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: train_word2vec
train_word2vec:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--cluster albedo \
	--properties 'spark.executor.memory=13312m,spark.jars.packages=com.hankcs:hanlp:portable-1.3.4,spark.albedo.dataDir=gs://albedo/spark-data' \
	--class ws.vinta.albedo.Word2VecRecommender \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,com.hankcs:hanlp:portable-1.3.4" \
	--class ws.vinta.albedo.Word2VecRecommender \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: train_ranker
train_ranker:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--cluster albedo \
	--properties 'spark.executor.memory=13312m,spark.jars.packages=com.databricks:spark-avro_2.11:3.2.0,spark.albedo.dataDir=gs://albedo/spark-data' \
	--class ws.vinta.albedo.PersonalizedRankerTrainer \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2" \
	--class ws.vinta.albedo.PersonalizedRankerTrainer \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: build_user_profile
build_user_profile:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--cluster albedo \
	--properties 'spark.executor.memory=13312m,spark.albedo.dataDir=gs://albedo/spark-data' \
	--class ws.vinta.albedo.UserProfileBuilder \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.UserProfileBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: build_repo_profile
build_repo_profile:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--cluster albedo \
	--properties 'spark.executor.memory=13312m,spark.albedo.dataDir=gs://albedo/spark-data' \
	--class ws.vinta.albedo.RepoProfileBuilder \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--driver-memory 4g \
	--executor-cores 4 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.RepoProfileBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar
endif
