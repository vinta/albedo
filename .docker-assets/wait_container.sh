#!/bin/bash

CONTAINER_NAME="$1"

until [ "`docker inspect -f {{.State.Running}} $CONTAINER_NAME`" == "true" ]; do
    sleep 1;
done;
