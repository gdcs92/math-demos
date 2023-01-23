import numpy as np
import matplotlib.pyplot as plt


class ClickHandler:

    default_epsilon = 5
    """default pixel distance tolerance"""

    def __init__(
            self, fig, ax, mX, mY, D, stemgraph_lines,
            epsilon=None
    ):
        self.fig = fig
        self.ax = ax
        # coordenadas dos pontos que podem ser movidos
        self.mX = mX
        self.mY = mY
        self.D = D # pontos x da curva (domínio)
        self.stemgraph_lines = stemgraph_lines
        self.epsilon = epsilon or self.default_epsilon # pixel distance tol

        self.pind = None # active point index


    def _init_movable_pts_display_coords(self):
        # coloca coordenadas dos pontos no grafico em formato conveniente
        xr = np.reshape(self.mX, (np.shape(self.mX)[0], 1))
        yr = np.reshape(self.mY, (np.shape(self.mY)[0], 1))
        xy_vals = np.append(xr, yr, 1)

        # transforma coordenadas do grafico em coordenadas do display
        self._xyt = self.ax.transData.transform(xy_vals)
        # xt, yt = xyt[:, 0], xyt[:, 1]

    def on_button_press(self, event):
        'whenever a mouse button is pressed'

        if event.inaxes is None or event.button != 1:
            return

        self.pind = self._get_ind_under_point(event)

    def _get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

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
        markerline = self.stemgraph_lines.markerline
        markerline.set_ydata(self.mY)

        stemlines = self.stemgraph_lines.stemlines
        n = self.pind
        segments = stemlines.get_segments()
        segments[n][1, 1] = self.mY[n]
        stemlines.set_segments(segments)

    def on_motion_notify(self, event):
        'on mouse movement'

        if self.pind is None or event.inaxes is None or event.button != 1:
            return

        # atualiza pontos movíveis
        self.mY[self.pind] = event.ydata
        self._update_stemline()

        self.fig.canvas.draw_idle()
