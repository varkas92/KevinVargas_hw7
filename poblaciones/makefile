n_steps = 20000
n_burn = 5000

all: mcmc_lotkavolterra.pdf

mcmc_lotkavolterra.pdf:
	python mcmc_lotkavolterra.py $(n_steps) $(n_burn)

clean:
	rm -f mcmc_lotkavolterra.pdf
