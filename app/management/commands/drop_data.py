# coding: utf-8

from django.core.management.base import BaseCommand

from app.models import RepoStarring
from app.models import UserRelation


class Command(BaseCommand):

    def handle(self, *args, **options):
        UserRelation.objects.all().delete()
        RepoStarring.objects.all().delete()
