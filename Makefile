.PHONY: clean
clean:
	find . -name \*.pyc -o -name \*.pyo -o -name __pycache__ -exec rm -rf {} +

.PHONY: up
up:
	mkdir -p ../albedo-dist-packages
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

.PHONY: shell
shell:
	docker exec -i -t albedo_django_1 python manage.py shell_plus

.PHONY: upload_db
upload_db:
	aws s3 cp albedo.sql s3://files.albedo.one/albedo.sql

.PHONY: download_db
download_db:
	aws s3 cp s3://files.albedo.one/albedo.sql albedo.sql

.PHONY: spark_standardalone
spark_standardalone:
	cd ${SPARK_HOME} && ./sbin/start-master.sh -h localhost
	cd ${SPARK_HOME} && ./sbin/start-slave.sh spark://localhost:7077
	open http://localhost:8080/

.PHONY: spark_stop
spark_stop:
	cd ${SPARK_HOME} && ./sbin/stop-master.sh
	cd ${SPARK_HOME} && ./sbin/stop-slave.sh

.PHONY: spark_shell
spark_shell:
	PYSPARK_DRIVER_PYTHON="jupyter" \
	PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip 0.0.0.0" \
	pyspark \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--driver-memory 4g \
	--executor-memory 15g \
	--master spark://localhost:7077

.PHONY: spark_submit
spark_submit:
	cd spark_app/src/main/python/deps/ && zip -x \*/__pycache__/\* -r ../deps.zip * && cd .. && \
	spark-submit \
	--packages "com.github.fommil.netlib:all:1.1.2,mysql:mysql-connector-java:5.1.41" \
	--driver-memory 4g \
	--executor-memory 15g \
	--master spark://localhost:7077 \
	--py-files deps.zip \
	train_als.py -- -u vinta

.PHONY: spark_submit_gcp
spark_submit_gcp:
	cd spark_app/src/main/python/deps/ && zip -x \*/__pycache__/\* -r ../deps.zip * && cd .. && \
	gcloud dataproc jobs submit pyspark \
	--cluster albedo \
	--py-files deps.zip \
	train_als.py -- -u vinta
