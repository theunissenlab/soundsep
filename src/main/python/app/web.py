import json
import requests
from functools import partial

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui as gui

from app.settings import read_default


class Selector(widgets.QFrame):
    selectEvent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()

    def set_url(self, base_url):
        self.base_url = base_url

    def init_ui(self):
        self.mainLayout = widgets.QVBoxLayout()
        self.selectorDropdown = widgets.QComboBox(self)  # dropdown of subjects

        self.scrollArea = widgets.QScrollArea()
        frame = widgets.QGroupBox()
        self.scrollArea.setWidget(frame)
        self.scrollArea.setWidgetResizable(True)

        self.optionsListLayout = widgets.QVBoxLayout()
        self.submitButton = widgets.QPushButton("Open")
        frame.setLayout(self.optionsListLayout)
        self.optionsListLayout.setSizeConstraint(gui.QLayout.SetMinAndMaxSize)
        self.scrollArea.setWidget(frame)

        self.mainLayout.addWidget(self.selectorDropdown)
        self.mainLayout.addWidget(self.scrollArea)
        # self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.submitButton)

        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)

        self.selectorDropdown.currentIndexChanged.connect(self.on_item_change)
        self.setLayout(self.mainLayout)

    def fill_dropdown(self, items):
        self.selectorDropdown.clear()
        for key in items:
            self.selectorDropdown.addItem(key, items[key])

    def on_item_change(self):
        selected_index = self.selectorDropdown.currentIndex()

        for i in reversed(range(self.optionsListLayout.count())):
            item = self.optionsListLayout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            else:
                self.optionsListLayout.removeItem(item)

        # load the url for that wdiget and fill the options List
        selected_item = self.selectorDropdown.itemData(selected_index)
        for item in selected_item:
            itemWidget = widgets.QWidget()
            layout = widgets.QGridLayout()
            selectButton = widgets.QPushButton("Select")
            selectButton.clicked.connect(partial(self.selectEvent.emit, item["path"]))
            selectButton.setDisabled(not item["has_lazy"])
            layout.addWidget(widgets.QLabel(item["name"]), 0, 0, 1, 2)
            label = widgets.QLabel("Ready" if item["has_lazy"] else "")
            label.setStyleSheet("color: green;");

            layout.addWidget(label, 0, 2, 1, 2)
            layout.addWidget(selectButton, 0, 4, 1, 1)
            itemWidget.setLayout(layout)
            hint = itemWidget.sizeHint()
            if hint.isValid():
                itemWidget.setMinimumSize(hint)
            itemWidget.setFixedHeight(50)
            self.optionsListLayout.addWidget(itemWidget)
        self.optionsListLayout.addStretch()


class WebLoader(widgets.QDialog):
    selectEvent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()

        self.chooseSessionWidget.selectEvent.connect(self._on_select)
        self.chooseSiteWidget.selectEvent.connect(self._on_select)

    def init_ui(self):
        self.mainLayout = widgets.QVBoxLayout()

        self.serverLayout = widgets.QHBoxLayout()
        self.chooseDataTabs = widgets.QTabWidget()
        self.chooseDataTabs.setDisabled(True)

        self.serverUrlInput = widgets.QLineEdit(self, placeholderText="Server Url")
        if read_default.SERVER_URL:
            self.serverUrlInput.setText(read_default.SERVER_URL)

        self.loadUrlButton = widgets.QPushButton("Connect")
        self.serverLayout.addWidget(self.serverUrlInput)
        self.serverLayout.addStretch()
        self.serverLayout.addWidget(self.loadUrlButton)
        self.loadUrlButton.clicked.connect(self.load_server_url)

        self.chooseSessionWidget = Selector(self)
        self.chooseSiteWidget = Selector(self)

        self.chooseDataTabs.addTab(self.chooseSessionWidget, "Sessions")
        self.chooseDataTabs.addTab(self.chooseSiteWidget, "Sites")

        self.mainLayout.addLayout(self.serverLayout)
        self.mainLayout.addWidget(self.chooseDataTabs)

        self.setLayout(self.mainLayout)
        self.setMinimumWidth(800)
        self.setMaximumHeight(800)
        self.setFixedHeight(600)

    def load_server_url(self):
        url = self.serverUrlInput.text()
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            widgets.QMessageBox.warning(
                self,
                "Error",
                "Could not connect to {}".format(url),
            )
            self.chooseDataTabs.setDisabled(True)
        else:
            # now look for sessions and sites
            response = requests.get(url + "/sessions")
            result = json.loads(response.text)
            self.chooseSessionWidget.fill_dropdown(result)

            response = requests.get(url + "/sites")
            result = json.loads(response.text)
            self.chooseSiteWidget.fill_dropdown(result)
            read_default.set("SERVER_URL", url)
            self.chooseDataTabs.setDisabled(False)

    def _on_select(self, path):
        url = self.serverUrlInput.text()
        self.selectEvent.emit(url + path)
