FROM continuumio/miniconda3
RUN conda install -c conda-forge fenics
COPY .git/ /tmp/mud-examples/.git
COPY src/ /tmp/mud-examples/src
COPY README.md /tmp/mud-examples
COPY MANIFEST.in /tmp/mud-examples
COPY setup.py /tmp/mud-examples
COPY setup.cfg /tmp/mud-examples

RUN cd /tmp/mud-examples && \
  pip install . && \
  rm -rf /tmp/*

# RUN cd /tmp && pip install mud_examples
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
CMD mud_run_all
