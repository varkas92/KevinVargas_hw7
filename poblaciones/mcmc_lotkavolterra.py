import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

obs_data = np.loadtxt("lotka_volterra_obs.dat")
t_obs = obs_data[:, 0]
x_obs = obs_data[:, 1]
y_obs = obs_data[:, 2]

def likelihood(x_obs, x_model, y_obs, y_model):
    
    chi_squared_x = (1.0 / 2.0) * sum((x_obs - x_model) ** 2)
    chi_squared_y = (1.0 / 2.0) * sum((y_obs - y_model) ** 2)
    
    return chi_squared_x + chi_squared_y

def x_prima(x, y, alpha, beta):
    
    return x * (alpha - beta * y)

def y_prima(x, y, gamma, delta):
    
    return -y * (gamma - delta * x)

def RK4(i, xi, yi, alpha, beta, gamma, delta):
    
    h = t_obs[i] - t_obs[i - 1]
    
    k_1_x = x_prima(xi, yi, alpha, beta)
    k_1_y = y_prima(xi, yi, gamma, delta)
    
    #Paso 1
    
    x1 = xi + (h / 2.0) * k_1_x
    y1 = yi + (h / 2.0) * k_1_y
    
    k_2_x = x_prima(x1, y1, alpha, beta)
    k_2_y = y_prima(x1, y1, gamma, delta)
    
    #Paso 2
    
    x2 = xi + (h / 2.0) * k_2_x
    y2 = yi + (h / 2.0) * k_2_y
    
    k_3_x = x_prima(x2, y2, alpha, beta)
    k_3_y = y_prima(x2, y2, gamma, delta)
    
    #Paso 3
    
    x3 = xi + h * k_3_x
    y3 = yi + h * k_3_y
    
    k_4_x = x_prima(x3, y3, alpha, beta)
    k_4_y = y_prima(x3, y3, gamma, delta)
    
    #Paso 4
    av_k_x = (1.0 / 6.0) * (k_1_x + 2.0 * k_2_x + 2.0 * k_3_x + k_4_x)
    av_k_y = (1.0 / 6.0) * (k_1_y + 2.0 * k_2_y + 2.0 * k_3_y + k_4_y)
    
    x_new = xi + h * av_k_x
    y_new = yi + h * av_k_y
    
    return x_new, y_new

#Runge-Kutta cuarto orden
def my_model(alpha, beta, gamma, delta):
    
    n = 96
    
    x_mod = np.ones(n)
    y_mod = np.ones(n)

    x_mod[0] = x_obs[0]
    y_mod[0] = y_obs[0]

    for i in range(1, n):
    
        x_mod[i], y_mod[i] = RK4(i, x_mod[i - 1], y_mod[i - 1], alpha, beta, gamma, delta)
        
    return x_mod, y_mod

alpha_walk = np.empty((0))
beta_walk = np.empty((0))
gamma_walk = np.empty((0))
delta_walk = np.empty((0))

l_walk = np.empty((0))

alpha_walk = np.append(alpha_walk, np.random.random())
beta_walk = np.append(beta_walk, np.random.random())
gamma_walk = np.append(gamma_walk, np.random.random())
delta_walk = np.append(delta_walk, np.random.random())

n_steps = int(sys.argv[1])
n_burn = int(sys.argv[2])

for i in range(0, n_steps):
    
    alpha_prime = np.random.normal(alpha_walk[i], 0.1)
    beta_prime = np.random.normal(beta_walk[i], 0.1)
    gamma_prime = np.random.normal(gamma_walk[i], 0.1)
    delta_prime = np.random.normal(delta_walk[i], 0.1) 

    x_init, y_init = my_model(alpha_walk[i], beta_walk[i], gamma_walk[i], delta_walk[i])
    x_prime, y_prime = my_model(alpha_prime, beta_prime, gamma_prime, delta_prime)
    
    l_init = likelihood(x_obs, x_init, y_obs, y_init)
    l_prime = likelihood(x_obs, x_prime, y_obs, y_prime)
    
    gam = -l_prime + l_init
    
    if(gam >= 0.0):
        
        alpha_walk = np.append(alpha_walk, alpha_prime)
        beta_walk = np.append(beta_walk, beta_prime)
        gamma_walk = np.append(gamma_walk, gamma_prime)
        delta_walk = np.append(delta_walk, delta_prime)
        
        l_walk = np.append(l_walk, l_prime)
        
    else:
        
        bet = np.random.random()
        alp = np.exp(gam)
        
        if(bet <= alp):
            
            alpha_walk = np.append(alpha_walk, alpha_prime)
            beta_walk = np.append(beta_walk, beta_prime)
            gamma_walk = np.append(gamma_walk, gamma_prime)
            delta_walk = np.append(delta_walk, delta_prime)

            l_walk = np.append(l_walk, l_prime)

        else:
            
            alpha_walk = np.append(alpha_walk, alpha_walk[i])
            beta_walk = np.append(beta_walk, beta_walk[i])
            gamma_walk = np.append(gamma_walk, gamma_walk[i])
            delta_walk = np.append(delta_walk, delta_walk[i])

            l_walk = np.append(l_walk, l_init)

max_likelihood_id = np.argmin(l_walk[n_burn:])
best_alpha = alpha_walk[max_likelihood_id]
best_beta = beta_walk[max_likelihood_id]
best_gamma = gamma_walk[max_likelihood_id]
best_delta = delta_walk[max_likelihood_id]
incertidumbre = np.sqrt(l_walk[max_likelihood_id]) / len(t_obs)

print "alpha =" + str(best_alpha)
print "beta = " + str(best_beta)
print "gamma = " + str(best_gamma)
print "delta = " + str(best_delta)
print "incertidumbre = " + str(incertidumbre)

with PdfPages("mcmc_lotkavolterra.pdf") as pdf:

    count, bins, ignored = plt.hist(alpha_walk, 20, normed=True)

    plt.title("Densidad $alpha$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.

    count, bins, ignored = plt.hist(beta_walk, 20, normed=True)

    plt.title("Densidad $beta$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.

    count, bins, ignored = plt.hist(gamma_walk, 20, normed=True)

    plt.title("Densidad $gamma$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.

    count, bins, ignored = plt.hist(delta_walk, 20, normed=True)

    plt.title("Densidad $delta$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.
