import matplotlib.pyplot as plt
import numpy as np
from soundsig.sound import plot_spectrogram


class MultiChannelPlotter(object):
    def __init__(
                self,
                data,
                panel_size=(5, 5),
                layout="horizontal",
            ):
        """Conveniently plot create parallel plots for multiple channels of data

        To use, initialize with the proper data array, where each element of
        `data` represents the signal on that channel. The define a function
        f(x, ax) which will be called for each element of `data`. The function
        should take x, which is one element of the data array, and
        ax, the matplotlib Axes object, and then perform the operants necessary
        to plot x on ax. Assign this function to the instance's _plot_fn
        method using MultiChannelPlotter.set_plot_fn(), then call
        MultiChannelPlotter.plot().

        To modify axes before plotting, use setup_fig() and then map_axes(fn)
        to apply a function to all axes in the figure.

        Example
        =======
        >>> data = np.random.normal(size=(3, 100))
        >>> plotter = MultiChannelPlotter(data, panel_size=(2, 1))
        >>> plotter.set_plot_fn(lambda x, ax: ax.plot(x))
        >>> plotter.plot()

        Then the
        Parameters
        ==========
        data : iterable
            Each element of data should be the data/plotting information
            for each channel. In the simplest case, is an array of shape (n, m),
            where n is the number of channels and m is the number of samples
        panel_size : (float, float) tuple
            Tuple of (width, height) of each panel in the output figure.
            Defaults to (5, 5)
        layout : 'horizontal' or 'vertical'
            Direction which to stack the muliple channels. Defaults to
            'horizontal'
        """
        self.data = data
        self.panel_size = panel_size

        # We can add 'square' as an option to find the most square-ish layout possible
        if layout not in ("horizontal", "vertical"):
            raise ValueError("layout must be either 'horizontal'"
                " or 'vertical'")

        self.layout = layout
        self.n_channels = len(data)
        self.setup_fig()

    def _plot_fn(self, data, ax):
        raise NotImplementedError

    def set_plot_fn(self, fn):
        self._plot_fn = fn

    def setup_fig(self):
        self.fig = plt.figure(figsize=self.panel_size)
        self.axes = []
        for i in range(self.n_channels):
            if self.layout == "horizontal":
                self.axes.append(self.fig.add_axes([i + 0.1, 0.1, 0.8, 0.8]))
            elif self.layout == "vertical":
                self.axes.append(self.fig.add_axes([0.1, i + 0.1, 0.8, 0.8]))
        self.axes = np.array(self.axes)
        self._is_setup = True

    def map_axes(self, fn, select=slice(None, None)):
        """Map a function onto all axes in the figure"""
        for ax in self.axes[select]:
            fn(ax)

    def plot(self):
        for i in range(self.n_channels):
            self._plot_fn(self.data[i], self.axes[i])
        return self.fig


class MultiSpecPlotter(MultiChannelPlotter):

    def __init__(
                self,
                data,
                panel_size=(5, 5),
                layout="horizontal",
                **spec_kwargs
            ):
        """Plot spectrograms of multiple channels of audio data

        Example
        =======
        >>> plotter = MultiChannelPlotter(
        ...     [(t_spec, f_spec, spec) for t_spec, f_spec, spec in data],
        ...     panel_size=(1, 4),
        ...     layout="horizontal")
        >>> plotter.plot()

        Parameters
        ==========
        data : iterable of (t_spec, f_spec, spec)
            Each element of data should be a tuple with three arrays: t_spec,
            f_spec, and spec. t_spec should be of shape (n,), where n is the
            number of time bins in the spectrogram. f_spec should be of shape
            (m,), where m is the number of frequency bins in the spectrogram.
            spec should be of shape (n, m).
        panel_size : (float, float) tuple
            Tuple of (width, height) of each panel in the output figure.
            Defaults to (5, 5)
        layout : 'horizontal' or 'vertical'
            Direction which to stack the muliple channels. Defaults to
            'horizontal'
        **spec_kwargs :
            Keyword arguments to be passed into soundsig.sound.plot_spectrogram
        """

        super().__init__(
            data,
            panel_size=panel_size,
            layout=layout,
        )
        self._spec_kwargs = spec_kwargs

    def _plot_fn(self, data, ax):
        t_spec, f_spec, spec = data
        plot_spectrogram(t_spec, f_spec, spec, ax=ax, **self._spec_kwargs)

    def plot(self):
        for i in range(self.n_channels):
            self._plot_fn(self.data[i], self.axes[i])
            self.axes[i].set_yticklabels([])
            self.axes[i].set_xticklabels([])
            self.axes[i].set_ylabel("")
            self.axes[i].set_xlabel("")
        plt.show()
        plt.close(self.fig)
