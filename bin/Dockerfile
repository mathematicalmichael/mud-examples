FROM continuumio/miniconda3
RUN conda install -c conda-forge fenics
RUN pip install mud_examples
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
WORKDIR /work
#RUN chown mudder:mudder -R 777 /work
CMD mud_run_all
