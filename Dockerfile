FROM vinta/python:3.5

MAINTAINER Vinta Chen <vinta.chen@gmail.com>

RUN apt-get update && \
    apt-get install -Vy \
    -o APT::Install-Recommends=false -o APT::Install-Suggests=false \
    build-essential \
    git-core \
    libmysqlclient-dev \
    mysql-client-5.7 \
    zlib1g-dev && \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN mkdir -p /app
WORKDIR /app

COPY .docker-assets/ /app/.docker-assets/
RUN echo "source /app/.docker-assets/django_bash_completion.sh" >> /root/.bashrc
