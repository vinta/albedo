Albedo
======

A practice of Recommender System for discovering GitHub repos you may like.

**Albedo** is a fictional character in Dan Simmons's [Hyperion Cantos](https://en.wikipedia.org/wiki/Hyperion_Cantos) series. Councilor Albedo is the TechnoCore's AI Advisor to the Hegemony of Man.

## Setup

```bash
$ git clone https://github.com/vinta/albedo.git
$ cd albedo
$ pip install -r requirements.txt
$ python manage.py migrate
```

## Usage

You need to create your own `GITHUB_PERSONAL_TOKEN` on [your GitHub settings page](https://help.github.com/articles/creating-an-access-token-for-command-line-use/).

```bash
# this step might take a few hours to complete depends on how many repos you starred and how many users you followed
$ python manage.py collect_data -t GITHUB_PERSONAL_TOKEN -u vinta
# or
$ wget https://s3-ap-northeast-1.amazonaws.com/files.albedo.one/db.sqlite3 -O db.sqlite3

# username: albedo
# password: hyperion
$ python manage.py runserver 0.0.0.0:8000
$ open http://127.0.0.1:8000/admin/

$ python manage.py train_user_cf -u vinta
$ python manage.py train_item_cf -u vinta
$ python manage.py train_content_based -u vinta

# you have to install GraphLab Create manually
# https://turi.com/download/install-graphlab-create.html
$ python manage.py train_graphlab -u vinta
```
