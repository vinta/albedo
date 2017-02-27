from django.contrib import admin

from app.models import RepoStarring
from app.models import UserRelation


class RepoStarringAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'repo_owner_username', 'repo_language', 'stargazers_count', 'forks_count', 'repo_updated_at')
    list_filter = ('repo_language',)
    search_fields = ['from_username', 'repo_full_name']


admin.site.register(RepoStarring, RepoStarringAdmin)


class UserRelationAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'from_username', 'to_username')
    list_filter = ('relation',)
    search_fields = ['from_username', 'to_username']


admin.site.register(UserRelation, UserRelationAdmin)
