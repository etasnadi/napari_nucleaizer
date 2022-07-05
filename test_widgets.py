import sys
import os
from pathlib import Path

from qtpy.QtWidgets import QApplication
from napari_nucleaizer import prediction_widget

from napari_nucleaizer.nucleaizer_widget import Nucleaizer
from napari_nucleaizer.prediction_widget import PredictionWidget
from napari_nucleaizer.training_widget import TrainingWidget
from napari_nucleaizer.model_selector_widget import ModelSelectorWidget
from qtpy.QtWidgets import QWidget, QScrollArea, QVBoxLayout

class ViewerMock():
    def __init__(self):
        super(ViewerMock, self).__init__()

    def open(sef, x):
        pass

    def add_labels(sef, x, name):
        pass

    def add_shapes(sef, x, face_color, edge_color, properties, text, name):
        pass

    def add_image(self, x, name):
        pass

def get_scrolled_parent(widget, title, min_size=800):
    parent = QWidget()
    parent.setWindowTitle(title)
    scroll = QScrollArea()
    if min_size is not None:
        scroll.setMinimumHeight(min_size)
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    parent_layout = QVBoxLayout()
    parent_layout.addWidget(scroll)
    parent.setLayout(parent_layout)
    return parent

def show_training_gui():
    global app
    training_widget = TrainingWidget()
    main_window = get_scrolled_parent(training_widget, 'NucleAIzer training GUI')
    training_widget.onClicedkOpenProject()
    main_window.show()

    widget = get_scrolled_parent(PredictionWidget(Path('/home/ervin/.nucleaizer')), 'NucleAIzer prediction GUI', min_size=None)
    widget.show()

    app.exec_()

def show_prediction_gui():
    global app
    widget = PredictionWidget(Path('/home/ervin/.nucleaizer'))
    widget.show()
    app.exec_()

def show_main_gui():
    global app
    viewer = ViewerMock()
    widget = Nucleaizer(viewer)
    widget.show()
    app.exec_()

def show_model_selector_gui():
    global app
    widget = ModelSelectorWidget()
    widget.show()
    app.exec_()

def main():
    show_main_gui()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()