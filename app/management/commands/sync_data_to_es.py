# coding: utf-8

from django.core.management.base import BaseCommand

from app.mappings import RepoInfoDoc
from app.models import RepoInfo


class Command(BaseCommand):

    def handle(self, *args, **options):
        def batch_qs(qs, batch_size=500):
            total = qs.count()
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                yield (start, end, total, qs[start:end])

        large_qs = RepoInfo.objects.filter(stargazers_count__gte=10, stargazers_count__lte=290000, fork=False)
        for start, end, total, qs_chunk in batch_qs(large_qs):
            documents = []
            for repo_info in qs_chunk:
                repo_info_doc = RepoInfoDoc()
                repo_info_doc.meta.id = repo_info.id
                repo_info_doc.owner_id = repo_info.owner_id
                repo_info_doc.owner_username = repo_info.owner_username
                repo_info_doc.owner_type = repo_info.owner_type
                repo_info_doc.name = repo_info.name
                repo_info_doc.full_name = repo_info.full_name
                repo_info_doc.description = repo_info.description
                repo_info_doc.language = repo_info.language
                repo_info_doc.created_at = repo_info.created_at
                repo_info_doc.updated_at = repo_info.updated_at
                repo_info_doc.pushed_at = repo_info.pushed_at
                repo_info_doc.homepage = repo_info.homepage
                repo_info_doc.size = repo_info.size
                repo_info_doc.stargazers_count = repo_info.stargazers_count
                repo_info_doc.forks_count = repo_info.forks_count
                repo_info_doc.subscribers_count = repo_info.subscribers_count
                repo_info_doc.fork = repo_info.fork
                repo_info_doc.has_issues = repo_info.has_issues
                repo_info_doc.has_projects = repo_info.has_projects
                repo_info_doc.has_downloads = repo_info.has_downloads
                repo_info_doc.has_wiki = repo_info.has_wiki
                repo_info_doc.has_pages = repo_info.has_pages
                repo_info_doc.open_issues_count = repo_info.open_issues_count
                repo_info_doc.topics = repo_info.topics

                documents.append(repo_info_doc)

            RepoInfoDoc.bulk_save(documents)
