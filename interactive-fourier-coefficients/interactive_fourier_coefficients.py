import numpy as np
import matplotlib.pyplot as plt

from click_handler import ClickHandler
from utils import fourier_coef, cos, sin


sinusoidal = lambda x: cos(x, 3) + 0.9*sin(x, 5)


if __name__ == '__main__':

    num_coefficients = 30
    f = sinusoidal

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    X = np.linspace(-1, 1, 500)
    Y = f(X)
    L = X[-1] - X[0]
    N = list(range(num_coefficients))
    FN = np.array([fourier_coef(f, n, 0, L) for n in N])
    cos_coefs = FN[:, 0]
    sin_coefs = FN[:, 1]

    ax = axs[0]
    time_domain_line2d, = ax.plot(X, f(X), '-')
    ax.set_ylabel('$f(x)$')
    ax.grid()

    ax = axs[1]
    stemgraph_lines = ax.stem(N, cos_coefs, 'o', basefmt='C0')
    ax.set_ylabel('$a_n$')
    ax.set_xticks(N)
    ax.set_xlim(left=-0.5)
    ax.grid()

    click = ClickHandler(
        fig, axs, N, cos_coefs, sin_coefs, X, Y, stemgraph_lines,
        time_domain_line2d
    )

    fig.canvas.mpl_connect('button_press_event', click.on_button_press)
    fig.canvas.mpl_connect('button_release_event', click.on_button_release)
    fig.canvas.mpl_connect('motion_notify_event', click.on_motion_notify)

    plt.show()
