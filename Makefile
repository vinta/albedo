.PHONY: clean
clean:
	find . -name \*.pyc -o -name \*.pyo -o -name __pycache__ -exec rm -rf {} +

.PHONY: up
up:
	mkdir -p ../albedo-data
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

.PHONY: upload
upload:
	aws s3 cp db.sqlite3 s3://files.albedo.one/db.sqlite3

.PHONY: download
download:
	aws s3 cp s3://files.albedo.one/db.sqlite3 db.sqlite3

.PHONY: spark
spark:
	export PYSPARK_DRIVER_PYTHON="jupyter" && \
	export PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip 0.0.0.0" && \
	pyspark \
	--packages "org.xerial:sqlite-jdbc:3.16.1,mysql:mysql-connector-java:5.1.41,com.github.fommil.netlib:all:1.1.2" \
	--driver-memory 4g \
	--executor-memory 4g \
	--master "local[*]"
