FROM continuumio/miniconda3
RUN conda install -c conda-forge fenics
COPY . /tmp/mud-examples
RUN cd /tmp/mud-examples && \
  pip install .

# RUN cd /tmp && pip install mud_examples
RUN rm -rf /tmp/*
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=user

RUN addgroup \
  --gid $GROUP_ID \
  $USER_NAME
RUN adduser \
  --disabled-password \
  --gecos '' \
  --uid $USER_ID \
  --gid $GROUP_ID \
  $USER_NAME

WORKDIR /work
RUN chown -R $USER_NAME:$USER_NAME /work
USER $USER_NAME
CMD mud_run_all --version
