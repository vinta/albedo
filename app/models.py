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
    from_username = models.CharField(max_length=39)
    to_username = models.CharField(max_length=39)
    relation = models.CharField(max_length=16)

    class Meta:
        unique_together = (('from_username', 'relation', 'to_username'),)

    def __str__(self):
        return '@{0} {1} @{2}'.format(self.from_username, self.relation, self.to_username)

    @staticmethod
    @retry(retry_on_exception=retry_if_database_is_lock, **retry_kwargs)
    def create_one(from_username, relation, user_dict):
        ur = UserRelation()
        ur.from_username = from_username
        ur.relation = relation
        ur.to_username = user_dict['login']
        ur.save()

        return ur


class RepoStarring(models.Model):
    from_username = models.CharField(max_length=39)
    repo_owner_username = models.CharField(max_length=39)
    repo_owner_type = models.CharField(max_length=16)
    repo_name = models.CharField(max_length=100)
    repo_full_name = models.CharField(max_length=140)
    repo_url = models.URLField()
    repo_language = models.CharField(max_length=32)
    repo_description = models.CharField(max_length=191)
    repo_created_at = models.DateTimeField()
    repo_updated_at = models.DateTimeField()
    stargazers_count = models.PositiveIntegerField()
    forks_count = models.PositiveIntegerField()

    class Meta:
        unique_together = (('from_username', 'repo_full_name'),)

    def __str__(self):
        return '@{0} starred {1}'.format(self.from_username, self.repo_full_name)

    @staticmethod
    @retry(retry_on_exception=retry_if_database_is_lock, **retry_kwargs)
    def update_or_create_one(from_username, repo_dict):
        rs = RepoStarring()
        rs.from_username = from_username
        rs.repo_owner_username = repo_dict['owner']['login']
        rs.repo_owner_type = repo_dict['owner']['type']
        rs.repo_name = repo_dict['name']
        rs.repo_full_name = repo_dict['full_name']
        rs.repo_url = repo_dict['html_url']
        rs.repo_language = repo_dict['language'] if repo_dict['language'] else ''
        rs.repo_description = repo_dict['description'] if repo_dict['description'] else ''
        rs.repo_created_at = repo_dict['created_at']
        rs.repo_updated_at = repo_dict['updated_at']
        rs.stargazers_count = repo_dict['stargazers_count']
        rs.forks_count = repo_dict['forks_count']
        try:
            rs.save()
        except IntegrityError:
            RepoStarring.objects \
                .filter(from_username=rs.from_username, repo_full_name=rs.repo_full_name) \
                .update(repo_updated_at=rs.repo_updated_at, stargazers_count=rs.stargazers_count, forks_count=rs.forks_count)
