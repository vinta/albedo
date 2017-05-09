# coding: utf-8

from django.db import IntegrityError
from django.db import models

from retrying import retry


retry_kwargs = {
    'stop_max_attempt_number': 10,
    'wait_random_min': 100, 'wait_random_max': 2000,
}


def retry_if_database_is_lock(exc):
    from django.db import OperationalError
    if isinstance(exc, OperationalError):
        if str(exc) == 'database is locked':
            return True
    return False


class UserRelation(models.Model):
    from_user_id = models.IntegerField()
    from_username = models.CharField(max_length=39)
    to_user_id = models.IntegerField()
    to_username = models.CharField(max_length=39)
    relation = models.CharField(max_length=16)

    class Meta:
        unique_together = (('from_user_id', 'relation', 'to_user_id'),)

    def __str__(self):
        return '@{0} {1} @{2}'.format(self.from_username, self.relation, self.to_username)

    @staticmethod
    @retry(retry_on_exception=retry_if_database_is_lock, **retry_kwargs)
    def create_one(from_user, relation, to_user):
        ur = UserRelation()
        try:
            ur.from_user_id = from_user['id']
            ur.from_username = from_user['login']
            ur.relation = relation
            ur.to_user_id = to_user['id']
            ur.to_username = to_user['login']
        except KeyError:
            print('KeyError')
            print(from_user)
        ur.save()

        return ur


class RepoStarring(models.Model):
    from_user_id = models.IntegerField()
    from_username = models.CharField(max_length=39)
    repo_owner_id = models.IntegerField()
    repo_owner_username = models.CharField(max_length=39)
    repo_owner_type = models.CharField(max_length=16)
    repo_id = models.IntegerField()
    repo_name = models.CharField(max_length=100)
    repo_full_name = models.CharField(max_length=140)
    repo_url = models.URLField()
    repo_language = models.CharField(max_length=32)
    repo_description = models.TextField(max_length=191)
    repo_created_at = models.DateTimeField()
    repo_updated_at = models.DateTimeField()
    stargazers_count = models.IntegerField()
    forks_count = models.IntegerField()

    class Meta:
        unique_together = (('from_user_id', 'repo_id'),)

    def __str__(self):
        return '@{0} starred {1}'.format(self.from_username, self.repo_full_name)

    @staticmethod
    @retry(retry_on_exception=retry_if_database_is_lock, **retry_kwargs)
    def update_or_create_one(from_user, repo_dict):
        rs = RepoStarring()
        try:
            rs.from_user_id = from_user['id']
            rs.from_username = from_user['login']
            rs.repo_owner_id = repo_dict['owner']['id']
            rs.repo_owner_username = repo_dict['owner']['login']
            rs.repo_owner_type = repo_dict['owner']['type']
            rs.repo_id = repo_dict['id']
            rs.repo_name = repo_dict['name']
            rs.repo_full_name = repo_dict['full_name']
            rs.repo_url = repo_dict['html_url']
            rs.repo_language = repo_dict['language'] if repo_dict['language'] else ''
            rs.repo_description = repo_dict['description'] if repo_dict['description'] else ''
            rs.repo_created_at = repo_dict['created_at']
            rs.repo_updated_at = repo_dict['updated_at']
            rs.stargazers_count = repo_dict['stargazers_count']
            rs.forks_count = repo_dict['forks_count']
        except KeyError:
            print('KeyError')
            print(from_user)
        try:
            rs.save()
        except IntegrityError:
            RepoStarring.objects \
                .filter(from_user_id=rs.from_user_id, repo_id=rs.repo_id) \
                .update(repo_updated_at=rs.repo_updated_at, stargazers_count=rs.stargazers_count, forks_count=rs.forks_count)
