import numpy as np
import matplotlib.pyplot as plt

from utils import fourier_coef, cos, sin


class ClickHandler:

    default_epsilon = 5
    """default pixel distance tolerance"""

    def __init__(
            self, fig, axs, N, cos_coefs, sin_coefs, X, Y,
            time_domain_line2d,
            cos_stemgraph_lines,
            sin_stemgraph_lines,
            epsilon=None
    ):
        self.fig = fig
        self.axs = axs
        # coordenadas dos pontos que podem ser movidos
        self.N = N
        self.cos_coefs = cos_coefs
        self.sin_coefs = sin_coefs
        self.X = X # pontos x da curva (domínio)
        self.Y = Y # pontos y da curva (domínio)
        self.cos_stemgraph_lines = cos_stemgraph_lines
        self.sin_stemgraph_lines = sin_stemgraph_lines
        self.time_domain_line2d = time_domain_line2d
        self.epsilon = epsilon or self.default_epsilon # pixel distance tol

        self.pind = None # active point index


    def _init_movable_pts_display_coords(self):
        # coloca coordenadas dos pontos no grafico em formato conveniente
        xr = np.reshape(self.N, (np.shape(self.N)[0], 1))
        yr = np.reshape(self.cos_coefs, (np.shape(self.cos_coefs)[0], 1))
        xy_vals = np.append(xr, yr, 1)

        # transforma coordenadas do grafico em coordenadas do display
        ax = self.axs[1]
        self._xyt = ax.transData.transform(xy_vals)
        # xt, yt = xyt[:, 0], xyt[:, 1]

    def on_button_press(self, event):
        'whenever a mouse button is pressed'

        if event.inaxes is None or event.button != 1:
            return

        self.pind = self._get_ind_under_point(event)

    def _get_axis_index(self, event):
        for i, ax in enumerate(self.axs):
            if event.inaxes == ax:
                return i
        else:
            return None

    def _get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        axis_index = self._get_axis_index(event)
        if axis_index != 1:
            return None

        self._init_movable_pts_display_coords()
        xt = self._xyt[:, 0]
        yt = self._xyt[:, 1]

        # encontra distâncias dos pontos do grafico ao ponto clicado
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_release(self, event):
        'whenever a mouse button is released'

        if event.button != 1:
            return

        self.pind = None

    def _update_stemline(self):
        markerline = self.cos_stemgraph_lines.markerline
        markerline.set_ydata(self.cos_coefs)

        stemlines = self.cos_stemgraph_lines.stemlines
        n = self.pind
        segments = stemlines.get_segments()
        segments[n][1, 1] = self.cos_coefs[n]
        stemlines.set_segments(segments)

    def _update_graph(self):
        num_coefs = len(self.N)
        A = self.cos_coefs
        B = self.sin_coefs
        L = self.X[-1] - self.X[0]

        def f(x):
            return (
                A[0]/2
                + sum(
                    A[n]*cos(x/L, n) + B[n]*sin(x/L, n)
                    for n in range(1, num_coefs)
                )
            )

        Y_coefs = f(self.X)

        self.Y = Y_coefs
        self.time_domain_line2d.set_ydata(self.Y)

    def on_motion_notify(self, event):
        'on mouse movement'

        if self.pind is None or event.inaxes is None or event.button != 1:
            return

        # atualiza pontos movíveis
        self.cos_coefs[self.pind] = event.ydata
        self._update_stemline()
        self._update_graph()

        self.fig.canvas.draw_idle()
