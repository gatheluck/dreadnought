FROM nvidia/cuda:11.0-devel-ubuntu20.04
ARG SCRATCH_DIRECTORY=/root/scratch

WORKDIR ${SCRATCH_DIRECTORY}
RUN apt update && apt install --no-install-recommends -y \
	git curl ssh openssh-client\
	python3 python3-pip \
	&& pip3 install poetry

RUN git clone https://github.com/gatheluck/dreadnought.git

WORKDIR ${SCRATCH_DIRECTORY}/dreadnought
RUN poetry install
RUN ln -s /var/data data
RUN mkdir output && ln -s /var/output output