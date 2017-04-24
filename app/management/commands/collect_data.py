# coding: utf-8

from concurrent.futures import ThreadPoolExecutor
import logging
import re
import time

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import IntegrityError

from retrying import retry
import requests

from app.models import UserRelation
from app.models import RepoStarring
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

    def __init__(self, token):
        self.worker_number = 10
        logger.info('worker_number: {0}'.format(self.worker_number))

        self.token = token

        self.min_stargazers_count = 30

        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'Albedo 0.2.0',
            'Authorization': 'token {0}'.format(self.token),
        }

    @retry(retry_on_exception=retry_if_remote_disconnected, wait_fixed=1000 * 60)
    def _make_reqeust(self, method, url, **kwargs):
        logger.info('make_reqeust: {0} {1}'.format(method, url))
        res = self.session.request('GET', url, **kwargs)

        if res.status_code == 403:
            # https://developer.github.com/v3/#rate-limiting
            if 'API rate limit exceeded' in res.json().get('message'):
                logger.info('Wait 15 minutes before retrying')
                time.sleep(60 * 15)
                res = self.session.request('GET', url, **kwargs)

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
            return res.json()

        with ThreadPoolExecutor(max_workers=self.worker_number) as executor:
            response_gen = executor.map(_fetch_page, range(1, total_page + 1))

        return response_gen

    def fetch_user_info(self, username):
        endpoint = 'https://api.github.com/users/{0}'.format(username)
        res = self._make_reqeust('GET', endpoint)
        return res.json()

    @timing_decorator
    def fetch_followed_users(self, username, fetch_more):
        from_user = self.fetch_user_info(username)

        try:
            UserRelation.create_one(from_user, 'be', from_user)
        except IntegrityError:
            pass

        endpoint = 'https://api.github.com/users/{0}/following'.format(username)
        for user_list in self._fetch_pages_concurrently(endpoint):
            for to_user in user_list:
                try:
                    UserRelation.create_one(from_user, 'followed', to_user)
                except IntegrityError:
                    continue
                if fetch_more:
                    self.fetch_followed_users(to_user, fetch_more=False)

    @timing_decorator
    def fetch_starred_repos(self, username):
        from_user = self.fetch_user_info(username)

        endpoint = 'https://api.github.com/users/{0}/starred'.format(username)
        for repo_list in self._fetch_pages_concurrently(endpoint):
            for repo in repo_list:
                # these situations could happen!
                if not isinstance(repo, dict):
                    continue
                if not repo.get('owner'):
                    continue

                if repo.get('stargazers_count', 0) <= self.min_stargazers_count:
                    continue

                RepoStarring.update_or_create_one(from_user, repo)

                if repo['owner']['type'] == 'User':
                    try:
                        UserRelation.create_one(from_user, 'starred', repo['owner'])
                    except IntegrityError:
                        pass


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('-t', '--token', action='store', dest='token', required=True)
        parser.add_argument('-u', '--username', action='store', dest='username', required=True)

    def handle(self, *args, **options):
        try:
            User.objects.create_superuser('albedo', email='', password='hyperion')
        except IntegrityError:
            pass

        github_token = options['token']
        github_username = options['username']

        self.stdout.write(self.style.SUCCESS('GitHub token: {0}'.format(github_token)))
        self.stdout.write(self.style.SUCCESS('GtiHub username: @{0}'.format(github_username)))

        self.stdout.write(self.style.SUCCESS('Start data collection'))

        crawler = GitHubCrawler(token=github_token)
        crawler.fetch_followed_users(github_username, fetch_more=True)
        crawler.fetch_starred_repos(github_username)

        related_usernames = UserRelation.objects \
            .filter(from_username=github_username) \
            .order_by('to_username') \
            .values_list('to_username', flat=True) \
            .distinct()
        user_count = related_usernames.count()
        self.stdout.write(self.style.SUCCESS('Total number of related users: {0}'.format(user_count)))

        for username in related_usernames.iterator():
            if username != github_username:
                crawler.fetch_starred_repos(username)

        repositories = RepoStarring.objects \
            .values_list('repo_full_name', flat=True) \
            .distinct()
        repo_count = repositories.count()
        self.stdout.write(self.style.SUCCESS('Total number of repositories: {0}'.format(repo_count)))
