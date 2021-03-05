# syntax = docker/dockerfile:experimental
FROM nvidia/cuda:11.0-devel-ubuntu20.04
ARG SCRATCH_DIRECTORY=/root/scratch
ARG SSH_PUBLICKEY=~/.ssh/github

WORKDIR ${SCRATCH_DIRECTORY}
RUN apt update && apt install --no-install-recommends -y \
	git curl ssh openssh-client\
	python3 python3-pip \
	&& pip3 install poetry

RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
COPY ${SSH_PUBLICKEY} /root/.ssh/
RUN git clone git@github.com:gatheluck/dreadnought.git
RUN rm -rf /root/.ssh/id_rsa

WORKDIR ${SCRATCH_DIRECTORY}/dreadnought
RUN poetry install