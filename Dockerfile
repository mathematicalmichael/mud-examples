FROM continuumio/miniconda3
RUN conda install -c conda-forge fenics
RUN pip install mud_examples
CMD ['mud_run_all']