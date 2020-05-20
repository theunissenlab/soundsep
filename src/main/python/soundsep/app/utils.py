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


class ThresholdAdjuster(object):
    """
    Some of the functions for selecting vocal intervals involves increasing
    or decreasing a threshold in a given range, or increasing or decreasing
    the fuse duration (for declaring nearby threshold crossings part of the
    same vocalizations).

    These values are not saved for every interval and can change depending
    on what range of data the user is currently selecting. So these are some
    helper functions that read in an amplitude envelope and known intervals
    (by index), estimate what a reasonable threshold / fuse duration "would
    have been", and adjust it for you.
    """
    def __init__(self, t_arr, y_data, intervals):
        self.t_arr = t_arr  # time axis
        self.y_data = y_data  # y axis
        self.intervals = intervals  # dataframe with columns "t_start" and "t_stop"

    def estimate(self):
        """Give estimated points for going down or up"""
        endpoint_values = []
        interval_max_values = []
        all_selector = np.zeros(len(self.t_arr)).astype(np.bool)

        if not len(self.intervals):
            return

        for idx in np.arange(len(self.intervals)):
            t1, t2 = self.intervals.iloc[idx][["t_start", "t_stop"]]
            selector = (self.t_arr <= t2) & (self.t_arr >= t1)
            all_selector = all_selector | selector
            endpoints = self.y_data[selector][[0, -1]]
            endpoint_values.append(endpoints[0])
            endpoint_values.append(endpoints[1])
            interval_max_values.append(np.max(self.y_data[selector]))

        bottom = np.mean(self.y_data[~all_selector])
        threshold = np.mean(endpoint_values)
        top = np.mean(interval_max_values)

        return 0.9 * threshold, threshold, 1.1 * threshold
