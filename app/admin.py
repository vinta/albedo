# coding: utf-8

from django.contrib import admin

from app.models import RepoInfo
from app.models import RepoStarring
from app.models import UserInfo
from app.models import UserRelation


class UserInfoAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'name', 'company', 'location', 'bio')
    search_fields = ['login', 'name', 'company']


admin.site.register(UserInfo, UserInfoAdmin)


class RepoInfoAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'language', 'stargazers_count', 'description')
    search_fields = ['full_name', 'description']


admin.site.register(RepoInfo, RepoInfoAdmin)


class UserRelationAdmin(admin.ModelAdmin):
    list_display = ('__str__', )
    list_filter = ('relation',)
    search_fields = ['from_username', 'to_username']


admin.site.register(UserRelation, UserRelationAdmin)


class RepoStarringAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'starred_at')
    search_fields = ['from_username', 'repo_full_name']


admin.site.register(RepoStarring, RepoStarringAdmin)
