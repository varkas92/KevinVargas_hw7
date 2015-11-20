import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

datos = np.loadtxt("monthrg.dat")

datos[:, 0] = datos[:, 0] + datos[:, 1] / 12 #Suma los meses a los annos.

index = np.where((datos[:, 3] != -99))[0]

nuevo = datos[index] #Archivo sin los datos desconocidos.

n_data = np.delete(nuevo, (1, 2, 4), 1) #Se eliminan todas las columnas que no se necesitan.

t_obs = n_data[:, 0] #Se toma el vector de tiempo en annos

y_obs = n_data[:, 1]

def likelihood(y_obs, y_model):
    
    chi_squared = (1.0 / 2.0) * sum((y_obs - y_model) ** 2)
    
    return chi_squared

def my_model(t_obs, a, b, c, d):
    
    return a * np.cos((2 * np.pi / d) * t_obs + b) + c

a_walk = np.empty((0))
b_walk = np.empty((0))
c_walk = np.empty((0))
d_walk = np.empty((0))

l_walk = np.empty((0))

a_walk = np.append(a_walk, np.random.random())
b_walk = np.append(b_walk, np.random.random())
c_walk = np.append(c_walk, np.random.random())
d_walk = np.append(d_walk, np.random.random())

n_steps = int(sys.argv[1])
n_burn = int(sys.argv[2])

for i in range(n_steps):
    
    a_prime = np.random.normal(a_walk[i], 0.1) 
    b_prime = np.random.normal(b_walk[i], 0.1)
    c_prime = np.random.normal(c_walk[i], 0.1)
    d_prime = np.random.normal(d_walk[i], 0.1)

    y_init = my_model(t_obs, a_walk[i], b_walk[i], c_walk[i], d_walk[i])
    y_prime = my_model(t_obs, a_prime, b_prime, c_prime, d_prime)
    
    l_prime = likelihood(y_obs, y_prime)
    l_init = likelihood(y_obs, y_init)
    
    gamma = -l_prime + l_init
    
    if(gamma >= 0.0):
        
        a_walk = np.append(a_walk, a_prime)
        b_walk = np.append(b_walk, b_prime)
        c_walk = np.append(c_walk, c_prime)
        d_walk = np.append(d_walk, d_prime)
        l_walk = np.append(l_walk, l_prime)
        
    else:
        
        beta = np.random.random()
        alpha = np.exp(gamma)
        
        if(beta <= alpha):
            
            a_walk = np.append(a_walk, a_prime)
            b_walk = np.append(b_walk, b_prime)
            c_walk = np.append(c_walk, c_prime)
            d_walk = np.append(d_walk, d_prime)
            l_walk = np.append(l_walk, l_prime)
            
        else:
            
            a_walk = np.append(a_walk, a_walk[i])
            b_walk = np.append(b_walk, b_walk[i])
            c_walk = np.append(c_walk, c_walk[i])
            d_walk = np.append(d_walk, d_walk[i])
            l_walk = np.append(l_walk, l_init)

max_likelihood_id = np.argmin(l_walk[n_burn:])
best_a = a_walk[max_likelihood_id]
best_b = b_walk[max_likelihood_id]
best_c = c_walk[max_likelihood_id]
best_d = d_walk[max_likelihood_id]

incertidumbre = np.sqrt(l_walk[max_likelihood_id]) / len(t_obs)

print "a = " + str(best_a)
print "b = " + str(best_b)
print "c = " + str(best_c)
print "d = " + str(best_d)
print "incertidumbre = " + str(incertidumbre)

with PdfPages("mcmc_solar.pdf") as pdf:

    count, bins, ignored = plt.hist(a_walk, 20, normed=True)

    plt.title("Densidad $a$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.

    count, bins, ignored = plt.hist(b_walk, 20, normed=True)

    plt.title("Densidad $b$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.

    count, bins, ignored = plt.hist(c_walk, 20, normed=True)

    plt.title("Densidad $c$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.

    count, bins, ignored = plt.hist(d_walk, 20, normed=True)

    plt.title("Densidad $d$")

    pdf.savefig() #Guarda la imagen en el pdf.
            
    plt.close() #Pasa a la siguiente pagina.
