#!/usr/bin/env bash

echo -e "Building PotatoTower Image"

DOCKER_BUILDKIT=1 \
docker build --pull --rm -f ./.docker/Dockerfile \
--build-arg BUILDKIT_INLINE_CACHE=1 \
--tag potato_tower:latest .