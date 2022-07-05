from pathlib import Path
from typing import Type

from qtpy.QtGui import QIcon, QFont, QShowEvent
from qtpy.QtCore import QSize, Signal, Slot
from qtpy.QtWidgets import (QStyle, QMessageBox, QDialog, QRadioButton, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QScrollArea)

from nucleaizer_backend.model_accessors import LocalModelAccessor, ZenodoModelList, ModelList

class FileChooser(QWidget):

    def __init__(self, lbl_text: str="Enter path...", input_text: str="Enter path...",  btn_text: str="...", parent: QWidget=None):
        super(FileChooser, self).__init__(parent)

        layout = QHBoxLayout()
        
        layout.addWidget(QLabel(lbl_text))
        
        self.path = QLineEdit(input_text)
        layout.addWidget(self.path)

        self.btn = QPushButton(btn_text)
        layout.addWidget(self.btn)
        self.btn.clicked.connect(self.open_browser)

        self.setLayout(layout)

    def open_browser(self):
        result = QFileDialog.getOpenFileName(self, "Select model!", '/home')[0]
        #result = QFileDialog.getExistingDirectory(self, "Select dataset directory", '/home', QFileDialog.ShowDirsOnly)

        if len(result) > 0 and Path(result).exists():
            self.path.setText(result)

class ModelSelectorDialog(QDialog):

    '''
    Selects a model from a list of models or from the file system.
    '''

    custom_accept_signal = Signal()

    def __init__(self, model_list: Type[ModelList], parent: QWidget=None):
        '''
        @arg model_list:
        '''
        super(ModelSelectorDialog, self).__init__()
        self.selected_model = None
        self.models_list_initialized = False

        self.model_list = model_list
        self.custom_model_path = None

        layout = QVBoxLayout()
        self.listing_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(600)
        scroll.setMinimumWidth(450)

        listing_widget = QWidget()
        listing_widget.setLayout(self.listing_layout)

        scroll.setWidget(listing_widget)

        self.custom_accept_signal.connect(self.accept)

        self.setWindowTitle("Segmentation profile selection")

        title = QLabel("Select the image that is similar to your data!")
        layout.addWidget(title)
        
        # Custom model selection from the file system
        section_label = QLabel("Custom model from the file system")
        section_label.setToolTip("Select this option if you have a model on your local filesystem.")
        section_label.setStyleSheet("QLabel { color : gray; }")
        
        self.listing_layout.addWidget(section_label)
        self.btncus_def_text = 'Custom'
        self.btncus = QRadioButton(self.btncus_def_text, self)
        self.listing_layout.addWidget(self.btncus)

        cancelbtn = QPushButton("Cancel")
        okbtn = QPushButton("OK")
        self.chooser = FileChooser(lbl_text="Model:", btn_text="browse...")
        self.chooser.setEnabled(False)
        self.chooser.path.textChanged.connect(self.custom_path_update)

        okbtn.setDefault(True)
        
        okbtn.clicked.connect(self.accept_)
        cancelbtn.clicked.connect(self.reject)
        self.btncus.toggled.connect(self.toggle_chooser)

        self.listing_layout.addWidget(self.chooser)

        layout.addWidget(scroll)
        layout.addWidget(okbtn)
        layout.addWidget(cancelbtn)

        self.setLayout(layout)

    def showEvent(self, a0: QShowEvent) -> None:
        if not self.models_list_initialized:
            self.show_models_list(self.model_list)
            self.models_list_initialized = True
        return super().showEvent(a0)

    def show_models_list(self, model_list):
        thumbw = 120
        thumbh = 120
        btnw = thumbw + 5
        btnh = thumbh + 5

        for idx, model in enumerate(model_list.get_models()):
            model_meta = model.get_meta()
            if len(model_meta['samples']) < 1:
                continue

            btn = QRadioButton(model_meta['name'].replace('&', '&&'), self)
            btn.setToolTip(model_meta['description'].replace('&', '&&'))
            btn.setIcon(QIcon(str(model.access_resource(model_meta['samples'][0]))))
            btn.setIconSize(QSize(thumbw, thumbh))
            btn.setMinimumSize(btnw, btnh)
            
            class Fn:
                def __init__(self, fun, btn, k):
                    self.fun = fun
                    self.btn = btn
                    self.k = k
                def __call__(self):
                    self.fun(self.btn, self.k)

            btn.toggled.connect(Fn(self.set_selected_from_list, btn, idx))

            #listing_layout.addWidget(btn)
            self.listing_layout.insertWidget(0, btn)

        section_label = QLabel(model_list.get_short_description())
        section_label.setToolTip(model_list.get_long_description())
        section_label.setStyleSheet("QLabel { color : gray; }")
        self.listing_layout.insertWidget(0, section_label)

    def accept_(self):
        if self.btncus.isChecked() :
            if self.custom_model_path is not None and self.custom_model_path.exists():
                try:
                    self.selected_model = LocalModelAccessor.from_path(self.custom_model_path)
                    self.custom_accept_signal.emit()
                except Exception as e:
                    msg = QMessageBox(self)
                    msg.setText("Error while loading the model: %s" % e)
                    msg.setWindowTitle("Error")
                    msg.exec_()
            else:
                msg = QMessageBox(self)
                msg.setText("Custom model selected but it (\"%s\") does not exist!" % str(self.custom_model_path))
                msg.setWindowTitle("Error")
                msg.exec_()
        else:
            if self.selected_model is not None:
                self.custom_accept_signal.emit()
            else:
                msg = QMessageBox(self)
                msg.setText("No model is selected!")
                msg.setWindowTitle("Error")
                msg.exec_()

    def custom_path_update(self, new_path: str):
        self.custom_model_path = Path(new_path)

    def set_selected_from_list(self, btn: QPushButton, idx: int) -> None:
        if btn.isChecked():
            self.selected_model = self.model_list.get_model(idx)

    def toggle_chooser(self):
        if self.btncus.isChecked():
            self.chooser.setEnabled(True)
        else:
            self.chooser.setEnabled(False)

    def get_selected(self):
        return self.selected_model

class ModelSelectorWidget(QWidget):
    '''
    A widget for model selection.

    This implementation contains a browse button only. 
    When the user clicks on the button, a modal dialog opens for model selection.
    After the user selected the model, the browse button will contain the icon and the name of the selected model. 
    '''

    model_selected_signal = Signal(object)
    download_progress_signal = Signal(int)

    def __init__(self, nucleaizer_home_path, parent: QWidget=None, select_model_text='Selected model: '):
        super(ModelSelectorWidget, self).__init__(parent)

        #models_list = LocalModelList(Path('/home/ervin/.nucleaizer/models'))
        model_list = ZenodoModelList(nucleaizer_home_path / 'zenodo_cache', callback=self.downnload_progress_callback)

        self.dial = ModelSelectorDialog(model_list)
        #self.dial.setModal(True)

        lay = QVBoxLayout()
        #lay.setSizeConstraint(QLayout.SetFixedSize)

        lay.addWidget(QLabel(select_model_text))

        model_widget = QWidget()
        laymodel = QHBoxLayout()

        btn = QPushButton("Browse for models...")
        self.btn = btn

        laymodel.addWidget(btn)
        model_widget.setLayout(laymodel)

        self.setLayout(lay)
        lay.addWidget(model_widget)
        btn.clicked.connect(self.click)

    def downnload_progress_callback(self, progress):
        self.download_progress_signal.emit(progress)

    def select_model(self, selected):
        selected_meta = selected.get_meta()

        self.btn.setText(selected_meta['name'].replace('&', '&&'))
        self.btn.setToolTip(selected_meta['description'].replace('&', '&&'))
        
        # Set icon if available, else set a simple file icon.
        if 'samples' in selected_meta.keys() and len(selected_meta['samples']) > 0:
            self.btn.setIcon(QIcon(str(selected.access_resource(selected_meta['samples'][0]))))
        else:
            pixmapi = getattr(QStyle, "SP_DriveHDIcon")
            icon = self.style().standardIcon(pixmapi)
            self.btn.setIcon(icon)
        self.btn.setIconSize(QSize(20, 20))
        self.model_selected_signal.emit(selected)

    def click(self):
        result = self.dial.exec()
        selected = self.dial.get_selected()
        if result > 0 and selected is not None:
            self.select_model(selected)