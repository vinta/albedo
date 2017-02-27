.PHONY: clean
clean:
	find . -regex "\(.*__pycache__.*\|*.py[co]\)" -delete

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

.PHONY: upload
upload:
	aws s3 cp db.sqlite3 s3://files.albedo.one/db.sqlite3
