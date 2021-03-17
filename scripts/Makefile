#REQUIRED_BINS := python jupyter-nbconvert
#$(foreach bin,$(REQUIRED_BINS),\
#    $(if $(shell command -v $(bin) 2> /dev/null ),$(info Found `$(bin)`),$(error Please install `$(bin)`)))

# DEPS:=mud_problem.py poisson.py

# estimated 2.5 minutes
# n is number of trials (repeatitions of experiment to avg out results)
# r stands for proportion of time elapsed
# t stands for tolerance (of measurement equipment)
ode:
	mud_run_ode
	@echo "Generated ./ode/"

# estimated four minutes
#pde1: res1u.pkl $(DEPS)
#	mkdir -p pde_1D/
#	./run_poisson.sh  # TODO: make this the 1d

# adding res2u.pkl below will generate all results.
pde2u:
	mud_run_pde -d u
	@echo "Generated files in ./pde_2D/"

pde2n:
	mud_run_pde -d n --prefix 95results
	@echo "Generated files in ./pde_2D/"

# estimated twenty minutes for 1000
res2u.pkl: generate_pde_data.sh
	generate_poisson_data -v -n 100 -d u -i 2 -p res

res2n.pkl: generate_pde_data.sh
	generate_poisson_data -v -n 100 -d n -i 2 -p res

# data: $(DATA)

lin/lin-dim-cov-convergence.png:
	mud_run_lin
	@echo "Generated ./lin/lin-dim-cov-convergence.png"

lin/lin-rank-cov-convergence.png:
	mud_run_lin
	@echo "Generated ./lin/lin-rank-cov-convergence.png"

ode/ode_20_reference_solution.png ode/ode_200_reference_solution.png ode/ode_convergence_mud_obs_mean.png ode/ode_convergence_mud_obs_var.png: ode

#pde_2D/pde_sensors_2D.png pde_2D/pde_in_2D.png pde_2D/pde_mud_2-1_N100.png pde_2D/pde_mud_2-2_N100.png: pde2u

pde_2D/map_surface.png \
pde_2D/mud_surface.png \
pde_2D/pde_convergence_obs_mean.png \
pde_2D/pde_convergence_obs_var.png \
pde_2D/ref_results2u_map_solutions_N100.png \
pde_2D/ref_results2u_map_solutions_N20.png \
pde_2D/ref_results2u_mud_initial_S1000.png \
pde_2D/ref_results2u_mud_solutions_N100.png \
pde_2D/ref_results2u_mud_solutions_N20.png: pde2u

contours/data_mismatch_contour.png \
contours/tikonov_contour.png \
contours/consistent_contour.png \
contours/classical_solution.png \
contours/consistent_solution.png:
	mud_run_lin

examples: \
	# linear-examples \
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

linear-examples: \
	lin/lin-dim-cov-convergence.png \
	lin/lin-rank-cov-convergence.png

pde-examples: \
	pde_2D/map_surface.png \
	pde_2D/mud_surface.png \
	pde_2D/pde_convergence_obs_mean.png \
	pde_2D/pde_convergence_obs_var.png \
	pde_2D/ref_results2u_map_solutions_N100.png \
	pde_2D/ref_results2u_map_solutions_N20.png \
	pde_2D/ref_results2u_mud_initial_S1000.png \
	pde_2D/ref_results2u_mud_solutions_N100.png \
	pde_2D/ref_results2u_mud_solutions_N20.png

clean:
	rm -f   \
	  lin/* \
	  ode/* \
	  pde_2D/*.png \
	  contours/*
	@echo "Cleaned up figures"


.PHONY: ode
