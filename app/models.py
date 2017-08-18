# coding: utf-8

from django.db import IntegrityError
from django.db import models

from django_mysql.models import ListTextField


class UserInfo(models.Model):
    login = models.CharField(max_length=39, unique=True)
    account_type = models.CharField(max_length=16)
    name = models.CharField(max_length=255)
    company = models.CharField(max_length=255, null=True, blank=True)
    blog = models.URLField(max_length=255, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    bio = models.CharField(max_length=160, null=True, blank=True)
    public_repos = models.IntegerField()
    public_gists = models.IntegerField()
    followers = models.IntegerField()
    following = models.IntegerField()
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    def __str__(self):
        return '@{0}'.format(self.login)

    @staticmethod
    def create_one(user_dict):
        user = UserInfo()
        try:
            user.id = user_dict['id']
            user.login = user_dict['login']
            user.account_type = user_dict['type']
            user.name = user_dict['name']
            user.company = user_dict['company']
            user.blog = user_dict['blog']
            user.location = user_dict['location']
            user.email = user_dict['email']
            user.bio = user_dict['bio']
            user.public_repos = user_dict['public_repos']
            user.public_gists = user_dict['public_gists']
            user.followers = user_dict['followers']
            user.following = user_dict['following']
            user.created_at = user_dict['created_at']
            user.updated_at = user_dict['updated_at']
        except (KeyError, TypeError) as e:
            print(e)
            print(user_dict)
            return

        try:
            user.save()
        except IntegrityError:
            pass


class RepoInfo(models.Model):
    owner_id = models.IntegerField()
    owner_username = models.CharField(max_length=39)
    owner_type = models.CharField(max_length=16)
    name = models.CharField(max_length=100)
    full_name = models.CharField(max_length=140, unique=True)
    description = models.TextField(max_length=191)
    language = models.CharField(max_length=32)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()
    pushed_at = models.DateTimeField()
    homepage = models.URLField(max_length=255, null=True, blank=True)
    size = models.IntegerField()
    stargazers_count = models.IntegerField()
    forks_count = models.IntegerField()
    subscribers_count = models.IntegerField()
    fork = models.BooleanField()
    has_issues = models.BooleanField()
    has_projects = models.BooleanField()
    has_downloads = models.BooleanField()
    has_wiki = models.BooleanField()
    has_pages = models.BooleanField()
    open_issues_count = models.IntegerField()
    topics = ListTextField(base_field=models.CharField(max_length=255))

    def __str__(self):
        return self.full_name

    @staticmethod
    def create_one(repo_dict):
        repo = RepoInfo()
        try:
            repo.id = repo_dict['id']
            repo.owner_id = repo_dict['owner']['id']
            repo.owner_username = repo_dict['owner']['login']
            repo.owner_type = repo_dict['owner']['type']
            repo.name = repo_dict['name']
            repo.full_name = repo_dict['full_name']
            repo.description = repo_dict['description']
            repo.language = repo_dict['language']
            repo.created_at = repo_dict['created_at']
            repo.updated_at = repo_dict['updated_at']
            repo.pushed_at = repo_dict['pushed_at']
            repo.homepage = repo_dict['homepage']
            repo.size = repo_dict['size']
            repo.subscribers_count = repo_dict['subscribers_count']
            repo.stargazers_count = repo_dict['stargazers_count']
            repo.forks_count = repo_dict['forks_count']
            repo.fork = repo_dict['fork']
            repo.has_issues = repo_dict['has_issues']
            repo.has_projects = repo_dict['has_projects']
            repo.has_downloads = repo_dict['has_downloads']
            repo.has_wiki = repo_dict['has_wiki']
            repo.has_pages = repo_dict['has_pages']
            repo.open_issues_count = repo_dict['open_issues_count']
            repo.topics = repo_dict['topics']
        except (KeyError, TypeError) as e:
            print(e)
            print(repo_dict)
            return

        try:
            repo.save()
        except IntegrityError:
            pass


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
    def create_one(from_user, relation, to_user):
        ur = UserRelation()
        try:
            ur.from_user_id = from_user['id']
            ur.from_username = from_user['login']
            ur.relation = relation
            ur.to_user_id = to_user['id']
            ur.to_username = to_user['login']
        except (KeyError, TypeError) as e:
            print(e)
            print(from_user)
            print(to_user)
            return

        try:
            ur.save()
        except IntegrityError:
            pass


class RepoStarring(models.Model):
    user_id = models.IntegerField()
    user_username = models.CharField(max_length=39)
    repo_id = models.IntegerField()
    repo_full_name = models.CharField(max_length=140)
    starred_at = models.DateTimeField()

    class Meta:
        unique_together = (('user_id', 'repo_id'),)

    def __str__(self):
        return '@{0} starred {1}'.format(self.user_username, self.repo_full_name)

    @staticmethod
    def create_one(user_dict, repo_dict):
        rs = RepoStarring()
        try:
            rs.user_id = user_dict['id']
            rs.user_username = user_dict['login']
            rs.repo_id = repo_dict['id']
            rs.repo_full_name = repo_dict['full_name']
            rs.starred_at = repo_dict['starred_at']
        except (KeyError, TypeError) as e:
            print(e)
            print(user_dict)
            print(repo_dict)
            return

        try:
            rs.save()
        except IntegrityError:
            pass
