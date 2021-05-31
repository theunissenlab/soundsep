class AppStateSingleton():
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class AppState(AppStateSingleton):
    def __init__(self):
        AppStateSingleton.__init__(self)
        if not hasattr(self, "_data"):
            self.reset()

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def update(self, update_dict):
        for key, val in update_dict.items():
            self._data[key] = val

    def has(self, key):
        return key in self._data

    def clear(self, key):
        if self.has(key):
            del self._data[key]

    def reset(self):
        self._data = {}


# There must be a nicer way to make another singleton object but this will do
class ViewStateSingleton():
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class ViewState(ViewStateSingleton):
    def __init__(self):
        ViewStateSingleton.__init__(self)
        if not hasattr(self, "_data"):
            self.reset()

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def update(self, update_dict):
        for key, val in update_dict.items():
            self._data[key] = val

    def has(self, key):
        return key in self._data

    def clear(self, key):
        if self.has(key):
            del self._data[key]

    def reset(self):
        self._data = {}
