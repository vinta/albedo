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
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
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

.PHONY: play
play:
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.Playground \
	target/albedo-1.0.0-SNAPSHOT.jar

.PHONY: baseline
baseline:
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.PopularityRecommenderTrainer \
	target/albedo-1.0.0-SNAPSHOT.jar

.PHONY: build_user_profile
build_user_profile:
ifeq ($(platform),gcp)
	# 5 min 18 sec
	time gcloud dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--class ws.vinta.albedo.UserProfileBuilder \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.UserProfileBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: build_repo_profile
build_repo_profile:
ifeq ($(platform),gcp)
	# 3 min 8 sec
	time gcloud dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--class ws.vinta.albedo.RepoProfileBuilder \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.RepoProfileBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: train_als
train_als:
ifeq ($(platform),gcp)
	# 10 min 19 sec
	time gcloud dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--class ws.vinta.albedo.ALSRecommenderBuilder \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.ALSRecommenderBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: cv_als
cv_als:
	time gcloud beta dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--jars target/albedo-1.0.0-SNAPSHOT.jar \
	--class ws.vinta.albedo.ALSRecommenderCV

.PHONY: build_cb
build_cb:
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,org.apache.httpcomponents:httpclient:4.5.2,org.elasticsearch.client:elasticsearch-rest-high-level-client:5.6.2,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.ContentRecommenderBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar

.PHONY: train_word2vec
train_word2vec:
ifeq ($(platform),gcp)
	# 38 min 58 sec
	time gcloud dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.jars.packages=com.hankcs:hanlp:portable-1.3.4;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--class ws.vinta.albedo.Word2VecCorpusBuilder \
	--jars target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,com.hankcs:hanlp:portable-1.3.4,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.Word2VecCorpusBuilder \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: train_lr
train_lr:
ifeq ($(platform),gcp)
	# 1 hr 35 min
	time gcloud beta dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.jars.packages=com.hankcs:hanlp:portable-1.3.4;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--jars target/albedo-1.0.0-SNAPSHOT.jar \
	--class ws.vinta.albedo.LogisticRegressionRanker
else
	time spark-submit \
	--verbose \
	--driver-memory 2g \
	--total-executor-cores 3 \
	--executor-cores 3 \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--packages "com.github.fommil.netlib:all:1.1.2,com.hankcs:hanlp:portable-1.3.4,mysql:mysql-connector-java:5.1.41" \
	--class ws.vinta.albedo.LogisticRegressionRanker \
	target/albedo-1.0.0-SNAPSHOT.jar
endif

.PHONY: cv_lr
cv_lr:
	time gcloud beta dataproc jobs submit spark \
	--verbosity debug \
	--cluster albedo \
	--properties "^;^spark.albedo.checkpointDir=gs://albedo/spark-data/checkpoint;spark.albedo.dataDir=gs://albedo/spark-data;spark.driver.memory=6g;spark.executor.cores=5;spark.executor.instances=4;spark.executor.memory=21g;spark.jars.packages=com.hankcs:hanlp:portable-1.3.4;spark.serializer=org.apache.spark.serializer.KryoSerializer" \
	--jars target/albedo-1.0.0-SNAPSHOT.jar \
	--class ws.vinta.albedo.LogisticRegressionRankerCV
