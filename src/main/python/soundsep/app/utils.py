import numpy as np

from app.settings import read_default


class TimeScrollManager(object):
    """
    Tool for working with breaking up the scrolling across time into discrete steps

    A view can be defined as a page index (integer) or by the time or time range
    that index represents.

    e.g. A 100s file, with step sizes of 1s, and window size of 10s

    There would be 100 pages: 0-10, 1-11, ..., 99-109
    The functions you want:
        page2time()
        time2page()

    Scrollbar deals in pages: you need to convert the page the scrollbar is on to
    a specific time to read.

    Vocal periods and detected intervals deal in times: to jump to a specific time
    you need to convert that time into a specific page

    Pages are represented as a tuple of (page index, t_start of page)

    """
    ALIGN_LEFT = "ALIGN_LEFT"
    ALIGN_RIGHT = "ALIGN_RIGHT"
    ALIGN_CENTER = "ALIGN_CENTER"

    def __init__(self, max_time):
        self.max_time = max_time

    @property
    def win_size(self):
        """Width of one window in seconds"""
        return read_default.WINDOW_SIZE

    @property
    def page_step(self):
        """Number of pages it takes to scroll one window"""
        return read_default.PAGE_STEP

    @property
    def page_size(self):
        """Size of a page in seconds"""
        return self.win_size / self.page_step

    def page2time(self, page):
        t1 = self.page_times()[page]
        t2 = min(t1 + self.win_size, self.max_time)
        return (t1, t2)

    def time2page(self, t, align="ALIGN_CENTER"):
        if align == self.ALIGN_LEFT:
            return int(np.floor(t / self.page_size))
        elif align == self.ALIGN_CENTER:
            t = t - 0.5 * self.win_size
            return int(np.floor(t / self.page_size))
        elif align == self.ALIGN_RIGHT:
            t = t - self.win_size
            return int(np.floor(t / self.page_size))
        elif isinstance(align, float):
            # Align the given time to the given screen fraction
            # i.e. t=3.0s should be aligned at align=0.4 of the screen from the
            # left.
            t = t - align * self.win_size
            if t < 0:
                t = 0
            elif t > self.max_time:
                t = self.max_time
            return int(np.floor(t / self.page_size))

    def set_max_time(self, max_time):
        self.max_time = max_time

    def pages(self):
        last_page_t = self.max_time - self.page_size
        n_pages = int(np.ceil(last_page_t / self.page_size))
        return np.arange(n_pages)

    def page_times(self):
        return self.pages() * self.page_size
