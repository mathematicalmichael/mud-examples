#REQUIRED_BINS := python jupyter-nbconvert
#$(foreach bin,$(REQUIRED_BINS),\
#    $(if $(shell command -v $(bin) 2> /dev/null ),$(info Found `$(bin)`),$(error Please install `$(bin)`)))

DEPS:=mud_problem.py poisson.py

# estimated 2.5 minutes
# n is number of trials (repeatitions of experiment to avg out results)
# r stands for proportion of time elapsed
# t stands for tolerance (of measurement equipment)
ode: mud_problem.py
	mkdir -p ode/
	./run_decay.sh

# estimated four minutes
pde1: res1u.pkl $(DEPS)
	mkdir -p pde_1D/
	./run_poisson.sh

# adding res2u.pkl below will generate all results.
pde2: MUD-Discretized.ipynb $(DEPS)
	mkdir -p pde_2D/
	jupyter-nbconvert --to notebook --execute \
	MUD-Discretized.ipynb && \
	rm -f MUD-Discretized.nbconvert.ipynb

# estimated twenty minutes
res2u.pkl: generate_pde_data.sh run_poisson.py poisson.py
	./generate_pde_data.sh

data: $(DATA)


lin/lin-dim-cov-convergence.png: MUD-Linear-Dimension.ipynb
	jupyter-nbconvert --to notebook --execute \
	MUD-Linear-Dimension.ipynb && \
	rm -f MUD-Linear-Dimension.nbconvert.ipynb

lin/lin-rank-cov-convergence.png: MUD-Linear-Rank.ipynb
	jupyter-nbconvert --to notebook --execute \
	MUD-Linear-Rank.ipynb && \
	rm -f MUD-Linear-Rank.nbconvert.ipynb

ode/ode_20_reference_solution.png ode/ode_200_reference_solution.png ode/ode_convergence_mud_obs_mean.png ode/ode_convergence_mud_obs_var.png: ode

pde_2D/pde_sensors_2D.png pde_2D/pde_in_2D.png pde_2D/pde_mud_2-1_N100.png pde_2D/pde_mud_2-2_N100.png: pde2

contours/data_mismatch_contour.png contours/tikonov_contour.png contours/consistent_contour.png contours/classical_solution.png contours/consistent_solution.png: Contours.ipynb
	jupyter-nbconvert --to notebook --execute \
	Contours.ipynb && \
	rm -f Contours.nbconvert.ipynb

examples: \
	linear-examples \
	ode-examples \
	contour-cartoons

contour-cartoons: \
	contours/data_mismatch_contour.png \
	contours/tikonov_contour.png \
	contours/consistent_contour.png \
	contours/classical_solution.png \
	contours/consistent_solution.png
	echo "Generated ./contours/"

ode-examples: \
	ode/ode_20_reference_solution.png \
	ode/ode_200_reference_solution.png \
	ode/ode_convergence_mud_obs_mean.png \
	ode/ode_convergence_mud_obs_var.png
	@echo "Generated ./ode/"

linear-examples: \
	lin/lin-dim-cov-convergence.png \
	lin/lin-rank-cov-convergence.png
	@echo "Generated ./lin/"

pde-examples: \
	pde_2D/pde_sensors_2D.png \
	pde_2D/pde_in_2D.png \
	pde_2D/pde_mud_2-1_N100.png \
	pde_2D/pde_mud_2-2_N100.png

clean:
	rm -f   \
	  lin/* \
	  ode/* \
	  pde_2D/*.png \
	  contours/*
	@echo "Cleaned up figures"


