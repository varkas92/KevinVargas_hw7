n_steps = 20000
n_burn = 5000

all: mcmc_solar.pdf

mcmc_solar.pdf:
	python mcmc_solar.py $(n_steps) $(n_burn)

clean:
	rm -f mcmc_solar.pdf
