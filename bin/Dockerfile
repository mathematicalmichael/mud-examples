FROM continuumio/miniconda3
RUN conda install -c conda-forge fenics
#COPY src/ /tmp/src
#COPY .git/ /tmp/.git/
#COPY README.md /tmp
#COPY MANIFEST.in /tmp
#COPY setup.py /tmp
#COPY setup.cfg /tmp
#RUN ls -al /tmp

RUN cd /tmp && pip install mud_examples
RUN rm -rf /tmp/*
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
WORKDIR /work
RUN chown -R user:user /work
USER user
CMD mud_run_all
