# coding: utf-8

from elasticsearch.helpers import bulk
from elasticsearch_dsl import analyzer
from elasticsearch_dsl import Date, Integer, Keyword, Text, Boolean
from elasticsearch_dsl import Index, DocType
from elasticsearch_dsl.connections import connections

client = connections.create_connection(hosts=['elasticsearch'])

repo_index = Index('repo')
repo_index.settings(
    number_of_shards=1,
    number_of_replicas=0
)

text_analyzer = analyzer(
    'text_analyzer',
    char_filter=["html_strip"],
    tokenizer="standard",
    filter=["asciifolding", "lowercase", "snowball", "stop"]
)
repo_index.analyzer(text_analyzer)


@repo_index.doc_type
class RepoInfoDoc(DocType):
    owner_id = Keyword()
    owner_username = Keyword()
    owner_type = Keyword()
    name = Text(text_analyzer, fields={'raw': Keyword()})
    full_name = Text(text_analyzer, fields={'raw': Keyword()})
    description = Text(text_analyzer)
    language = Keyword()
    created_at = Date()
    updated_at = Date()
    pushed_at = Date()
    homepage = Keyword()
    size = Integer()
    stargazers_count = Integer()
    forks_count = Integer()
    subscribers_count = Integer()
    fork = Boolean()
    has_issues = Boolean()
    has_projects = Boolean()
    has_downloads = Boolean()
    has_wiki = Boolean()
    has_pages = Boolean()
    open_issues_count = Integer()
    topics = Keyword(multi=True)

    class Meta:
        index = repo_index._name

    @classmethod
    def bulk_save(cls, documents):
        dicts = (d.to_dict(include_meta=True) for d in documents)
        return bulk(client, dicts)

    def save(self, **kwargs):
        return super(RepoInfoDoc, self).save(**kwargs)


RepoInfoDoc.init()
