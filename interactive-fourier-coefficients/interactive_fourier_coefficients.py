import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from click_handler import ClickHandler


PI = np.pi

def sin(x, k, phi=0): return np.sin(2*PI*(k*x + phi))

def cos(x, k, phi=0): return np.cos(2*PI*(k*x + phi))

def fourier_coef(f, n, xi, xf):
    L = xf - xi
    a_n = integrate.quad(lambda x: f(x)*cos(x/L, n), a=xi, b=xf)[0] * 2/L
    b_n = integrate.quad(lambda x: f(x)*sin(x/L, n), a=xi, b=xf)[0] * 2/L
    return a_n, b_n

sinusoidal = lambda x: cos(x, 3) + 0.9*sin(x, 5)


if __name__ == '__main__':

    num_coefficients = 30
    f = sinusoidal

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))

    X = np.linspace(-1, 1, 500)
    L = X[-1] - X[0]
    N = list(range(num_coefficients))
    FN = np.array([fourier_coef(f, n, 0, L) for n in N])
    cos_coefs = FN[:, 0]
    sin_coefs = FN[:, 1]

    stemgraph_lines = ax.stem(N, cos_coefs, 'o', basefmt='C0')
    ax.set_ylabel('$a_n$')
    ax.set_xticks(N)
    ax.set_xlim(left=-0.5)
    ax.grid()

    click = ClickHandler(
        fig, ax, N, cos_coefs, X, stemgraph_lines
    )

    fig.canvas.mpl_connect('button_press_event', click.on_button_press)
    fig.canvas.mpl_connect('button_release_event', click.on_button_release)
    fig.canvas.mpl_connect('motion_notify_event', click.on_motion_notify)

    plt.show()
