import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

PI = np.pi

def sin(x, k, phi=0): return np.sin(2*PI*(k*x + phi))

def cos(x, k, phi=0): return np.cos(2*PI*(k*x + phi))

def fourier_coef(f, n, xi, xf):
    L = xf - xi
    a_n = integrate.quad(lambda x: f(x)*cos(x/L, n), a=xi, b=xf)[0] * 2/L
    b_n = integrate.quad(lambda x: f(x)*sin(x/L, n), a=xi, b=xf)[0] * 2/L
    return a_n, b_n
