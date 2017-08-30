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
	cd src/main/python/deps/ && zip -r ../deps.zip * && cd .. && \
	PYSPARK_DRIVER_PYTHON="jupyter" \
	PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip 0.0.0.0" \
	pyspark \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--driver-memory 4g \
	--executor-memory 12g \
	--master spark://localhost:7077 \
	--py-files deps.zip

.PHONY: zeppelin_start
zeppelin_start:
	zeppelin-daemon.sh start
	open http://localhost:8080/
	open http://localhost:4040/jobs/

.PHONY: zeppelin_stop
zeppelin_stop:
	zeppelin-daemon.sh stop

.PHONY: train_als
train_als:
ifeq ($(platform),gcp)
	find . -name __pycache__ | xargs rm -Rf
	cd src/main/python/deps/ && zip -r ../deps.zip * && cd .. && \
	time gcloud dataproc jobs submit pyspark \
	--cluster albedo \
	--py-files deps.zip \
	train_als.py -- -u vinta
else
	find . -name __pycache__ | xargs rm -Rf
	cd src/main/python/deps/ && zip -r ../deps.zip * && cd .. && \
	time spark-submit \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--driver-memory 4g \
	--executor-memory 12g \
	--executor-cores 4 \
	--master spark://localhost:7077 \
	--py-files deps.zip \
	train_als.py -- -u vinta
endif

.PHONY: train_corpus
train_corpus:
ifeq ($(platform),gcp)
	time gcloud dataproc jobs submit spark \
	--class ws.vinta.albedo.GitHubCorpusTrainer \
	target/albedo-1.0.0-SNAPSHOT.jar
else
	time spark-submit \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--driver-memory 4g \
	--executor-memory 12g \
	--executor-cores 4 \
	--master spark://localhost:7077 \
	--class ws.vinta.albedo.GitHubCorpusTrainer \
	target/albedo-1.0.0-SNAPSHOT.jar
endif
