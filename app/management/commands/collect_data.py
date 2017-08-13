# coding: utf-8

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import random
import re
import time

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import IntegrityError

from retrying import retry
import requests

from app.models import RepoInfo
from app.models import RepoStarring
from app.models import UserInfo
from app.models import UserRelation
from app.utils_timing import timing_decorator


logger = logging.getLogger('django')


def retry_if_remote_disconnected(exc):
    from requests.exceptions import ConnectionError
    if isinstance(exc, ConnectionError):
        if 'RemoteDisconnected' in str(exc):
            logger.info('retry_if_remote_disconnected')
            return True
    return False


class GitHubCrawler(object):

    def __init__(self, tokens):
        self.tokens = tokens
        self.worker_number = 10
        self.min_stargazers_count = 1
        self.session = requests.Session()

        logger.info('worker_number: {0}'.format(self.worker_number))

    @property
    def random_token(self):
        return random.choice(self.tokens)

    @retry(retry_on_exception=retry_if_remote_disconnected, wait_fixed=1000 * 60)
    def _make_reqeust(self, method, url, **kwargs):
        logger.info('make_reqeust: {0} {1}'.format(method, url))

        headers = {
            'User-Agent': 'Albedo 1.0.0',
            'Accept': 'application/vnd.github.mercy-preview+json,application/vnd.github.v3.star+json',
            'Authorization': 'token {0}'.format(self.random_token),
        }
        res = self.session.request('GET', url, headers=headers, **kwargs)
        if res.status_code == 403:
            # https://developer.github.com/v3/#rate-limiting
            if 'API rate limit exceeded' in res.json().get('message'):
                logger.info('Wait 15 minutes before retrying')
                time.sleep(60 * 15)
                res = self.session.request('GET', url, headers=headers, **kwargs)

        return res

    def _parse_total_page(self, res):
        try:
            link = res.links['last']['url']
        except KeyError:
            total_page = 0
        else:
            try:
                page_number = re.search('https://api.github.com/[\w\/]+\?page=([\d]+)', link).group(1)
            except AttributeError:
                raise RuntimeError('Fail to parse the page number')
            total_page = int(page_number)

        logger.info('total_page: {0}'.format(total_page))
        return total_page

    def _fetch_pages_concurrently(self, endpoint):
        res = self._make_reqeust('GET', endpoint)
        total_page = self._parse_total_page(res)

        def _fetch_page(page_number):
            url = '{0}?page={1}'.format(endpoint, page_number)
            res = self._make_reqeust('GET', url, params={'page': page_number})
            try:
                content = res.json()
            except json.JSONDecodeError:
                content = []
            return content

        with ThreadPoolExecutor(max_workers=self.worker_number) as executor:
            response_gen = executor.map(_fetch_page, range(1, total_page + 1))

        return response_gen

    def fetch_user_info(self, username):
        endpoint = 'https://api.github.com/users/{0}'.format(username)
        res = self._make_reqeust('GET', endpoint)
        user_dict = res.json()
        UserInfo.create_one(user_dict)
        return user_dict

    def fetch_repo_info(self, repo_full_name):
        endpoint = 'https://api.github.com/repos/{0}'.format(repo_full_name)
        res = self._make_reqeust('GET', endpoint)
        repo_dict = res.json()
        RepoInfo.create_one(repo_dict)
        return repo_dict

    @timing_decorator
    def fetch_following_users(self, username, fetch_more):
        from_user = self.fetch_user_info(username)
        endpoint = 'https://api.github.com/users/{0}/following'.format(username)
        for user_list in self._fetch_pages_concurrently(endpoint):
            for to_user in user_list:
                UserRelation.create_one(from_user, 'followed', to_user)
                if fetch_more:
                    username = to_user['login']
                    self.fetch_following_users(username, fetch_more=False)

    @timing_decorator
    def fetch_follower_users(self, username, fetch_more):
        to_user = self.fetch_user_info(username)
        endpoint = 'https://api.github.com/users/{0}/followers'.format(username)
        for user_list in self._fetch_pages_concurrently(endpoint):
            for from_user in user_list:
                UserRelation.create_one(from_user, 'followed', to_user)
                if fetch_more:
                    username = from_user['login']
                    self.fetch_following_users(username, fetch_more=False)

    @timing_decorator
    def fetch_starred_repos(self, username):
        from_user = self.fetch_user_info(username)
        endpoint = 'https://api.github.com/users/{0}/starred'.format(username)
        for repo_list in self._fetch_pages_concurrently(endpoint):
            for starred in repo_list:
                # following situations could happen!
                if not isinstance(starred, dict):
                    continue
                repo = starred['repo']
                repo['starred_at'] = starred['starred_at']
                if repo.get('stargazers_count', 0) <= self.min_stargazers_count:
                    continue
                if not repo.get('owner'):
                    continue
                RepoStarring.create_one(from_user, repo)
                if repo['owner']['type'] == 'User':
                    UserRelation.create_one(from_user, 'starred', repo['owner'])


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('-t', '--tokens', type=lambda x: x.split(','), dest='tokens', required=True)
        parser.add_argument('-u', '--usernames', type=lambda x: x.split(','), dest='usernames', required=True)

    def handle(self, *args, **options):
        try:
            User.objects.create_superuser('albedo', email='', password='hyperion')
        except IntegrityError:
            pass

        github_tokens = options['tokens']
        github_usernames = options['usernames']

        self.stdout.write(self.style.SUCCESS('Start data collection'))
        crawler = GitHubCrawler(tokens=github_tokens)
        for github_username in github_usernames:
            self.stdout.write(self.style.SUCCESS('GtiHub username: @{0}'.format(github_username)))
            crawler.fetch_following_users(github_username, fetch_more=True)
            crawler.fetch_follower_users(github_username, fetch_more=False)
            crawler.fetch_starred_repos(github_username)

        from_usernames = UserRelation.objects \
            .values_list('from_username', flat=True) \
            .distinct()
        to_usernames = UserRelation.objects \
            .values_list('to_username', flat=True) \
            .distinct()
        usernames = set(from_usernames).union(set(to_usernames))
        user_count = len(usernames)
        self.stdout.write(self.style.SUCCESS('Total number of fetched users: {0}'.format(user_count)))

        for username in usernames:
            if username not in github_usernames:
                crawler.fetch_user_info(username)
                crawler.fetch_starred_repos(username)

        repositories = RepoStarring.objects \
            .values_list('repo_full_name', flat=True) \
            .distinct()
        repo_count = repositories.count()
        self.stdout.write(self.style.SUCCESS('Total number of fetched repositories: {0}'.format(repo_count)))

        for repo_full_name in repositories[:5]:
            crawler.fetch_repo_info(repo_full_name)

        self.stdout.write(self.style.SUCCESS('Done'))
