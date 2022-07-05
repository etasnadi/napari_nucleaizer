from pathlib import Path
import copy
import imageio

from napari.qt.threading import thread_worker
from qtpy.QtGui import QIcon, QFont, QShowEvent
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import (QWidget, QComboBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QListWidget, QCheckBox, QFileDialog, QMessageBox, QGroupBox, QListWidgetItem)

from napari_nucleaizer.model_selector_widget import ModelSelectorWidget
from . import common_stuff


class BatchPredictionWidget(QWidget):

    hide_batch_predict_widget = Signal()
    replace_quick_view_signal = Signal(tuple)
    reset_status_signal = Signal()
    update_progress_bar_signal = Signal(int)
    show_segmentation_signal = Signal(tuple)
    set_task_started_status_signal = Signal(str)
    open_image_signal = Signal(Path)

    IMAGE_PREDITC_EXTENSIONS = ['.tif', '.png', '.jpg']

    IMAGE_EXT = 'png'
    MASK_EXT = 'tiff'

    def __init__(self, nucleaizer_instance, parent=None):
        super(BatchPredictionWidget, self).__init__()
        self.parent = parent

        self.selected_input_dir = None
        self.selected_out_dir = None
        self.default_dir = '/home/ervin/nucleaizer_dataset'

        self.batch_predict_layout = QVBoxLayout()
        self.setLayout(self.batch_predict_layout)

        self.batch_predict_start_button = QPushButton("Batch predict selected")
        self.batch_predict_stop_button = QPushButton("Stop")
        self.input_directory_path = QLabel("Input dir")
        self.select_input_dir_button = QPushButton("Select...")
        self.output_directory_path = QLabel("Output dir")
        self.select_output_dir_button = QPushButton("Select...")
        self.input_images_list_widget = QListWidget()
        self.batch_predict_cancel_button = QPushButton("Cancel")
        
        self.default_selection = 2
        self.selection_cb = QComboBox()
        self.selection_cb.addItems(["All", "Prediction exists", "Prediction does not exist", "None"])

        self.batch_predict_layout.addWidget(QLabel("Predict selected"))
        
        widgets = [self.input_directory_path, self.select_input_dir_button, self.output_directory_path, self.select_output_dir_button, self.selection_cb,
            self.input_images_list_widget, self.batch_predict_start_button, self.batch_predict_stop_button, self.batch_predict_cancel_button]
        
        for widget in widgets:
            self.batch_predict_layout.addWidget(widget)

        self.selection_cb.setVisible(False)
        self.batch_predict_start_button.setEnabled(False)
        self.input_images_list_widget.setVisible(False)
        self.input_images_list_widget.setMinimumHeight(150)

        self.input_images_list_widget.itemDoubleClicked.connect(self.onDoubleClickedInputImage)
        self.input_images_list_widget.currentItemChanged.connect(self.onChangedInputImage)
        self.batch_predict_cancel_button.clicked.connect(self.onClickedBatchPredictCancel)
        self.batch_predict_start_button.clicked.connect(self.onClickStartBatchPredict)
        self.batch_predict_stop_button.clicked.connect(self.onClickStopBatchPredict)
        self.select_output_dir_button.clicked.connect(self.onClickSelectOutputDir)
        self.select_input_dir_button.clicked.connect(self.onClickSelectInputDir)
        self.selection_cb.currentIndexChanged.connect(self.selectionChanged)

        self.nucleaizer_instance = nucleaizer_instance

    def fill_input_images_list(self):
        self.input_images_list_widget.clear()
        
        for file in Path(self.selected_input_dir).iterdir():
            ext = ''

            if len(file.suffixes) > 0:
                ext = file.suffixes[-1]

            if ext in self.IMAGE_PREDITC_EXTENSIONS:
                item = QListWidgetItem(common_stuff.strip_text("%s" % file.name))
                item.setIcon(QIcon(str(file)))
                item.setData(Qt.UserRole, file)    
                self.input_images_list_widget.addItem(item)

    def batch_mark_completed_files(self):
        normal_font = QFont()
        bold_font = QFont()
        bold_font.setBold(True)

        n_files = self.input_images_list_widget.count()
        for item_idx in range(n_files):
            item = self.input_images_list_widget.item(item_idx)

            file = item.data(Qt.UserRole)

            _, mask = self.get_selected_input_instance(file)

            if mask is None:
                item.setFont(bold_font)
            else:
                item.setFont(normal_font)

    def activate_input_images_widget(self):
        self.select_input_dir_button.setText(common_stuff.strip_text(str(self.selected_input_dir.resolve()), 12))
        self.fill_input_images_list()
        self.batch_mark_completed_files()
        self.input_images_list_widget.setVisible(True)
        self.selection_cb.setVisible(True)
        self.selectionChanged(self.default_selection)
        self.selection_cb.setCurrentIndex(self.default_selection)

    def onClickSelectInputDir(self):
        selected_input_dir = QFileDialog.getExistingDirectory(
            self, "Select input directory", 
            str(self.default_dir), QFileDialog.ShowDirsOnly)

        if selected_input_dir is None:
            return

        selected_input_dir = Path(selected_input_dir)

        print('Selected input directory:', selected_input_dir, 'Exists?', selected_input_dir.exists())

        if selected_input_dir.exists():
            self.selected_input_dir = selected_input_dir
            self.activate_input_images_widget()
        
        if self.selected_input_dir is not None and self.selected_out_dir is not None:
            self.batch_predict_start_button.setEnabled(True)

    def onClickSelectOutputDir(self):
        self.selected_out_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory", 
            str(self.default_dir), QFileDialog.ShowDirsOnly)

        self.select_output_dir_button.setText(common_stuff.strip_text(self.selected_out_dir, 12))
        
        if self.selected_input_dir is not None:
            self.activate_input_images_widget()

        if self.selected_input_dir is not None and self.selected_out_dir is not None:
            self.batch_predict_start_button.setEnabled(True)

    def onClickStopBatchPredict(self):
        print('Stop prediction!', self.worker.quit())

    def onClickStartBatchPredict(self):
        print('Batch predicting...')
        
        cellsize_widget = self.parentWidget().mean_size_widget

        if cellsize_widget.get_cellsize_enabled():
            cellsize_ = cellsize_widget.get_cellsize()

        images_list = self.get_selected_file_list()

        self.set_task_started_status_signal.emit('Batch predicting images...')

        self.worker = self.run_segmentation_batch(images_list, cellsize_)
        self.worker.returned.connect(self.onReturnedBatchSegmentation)
        self.worker.yielded.connect(self.onYieldedBatchSegmentation)
        self.worker.start()

    def update_selection(self, fun):
        n_selected = 0
        n_files = self.input_images_list_widget.count()
        for item_idx in range(n_files):
            item = self.input_images_list_widget.item(item_idx)
            file = item.data(Qt.UserRole)
            if fun(file):
                n_selected += 1
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def get_selected_file_list(self):
        selected_items = []
        for item_idx in range(self.input_images_list_widget.count()):
            item = self.input_images_list_widget.item(item_idx)
            if item.checkState() == Qt.Checked:
                selected_items.append(item.data(Qt.UserRole))
        return selected_items

    def selectionChanged(self, i):
        print('Selected row:', i)
        selection_functions = [
            lambda x: True, 
            lambda x: self.get_selected_input_instance(x)[1] is not None,
            lambda x: self.get_selected_input_instance(x)[1] is None,
            lambda x: False]
        self.update_selection(selection_functions[i])

    def onChangedInputImage(self, curr, prev):
        selected_file = curr.data(Qt.UserRole)

        to_display = [selected_file, None]

        if self.selected_out_dir is not None:
            to_display = self.get_selected_input_instance(selected_file)
    
        if to_display[1] == None:
            self.replace_quick_view_signal.emit((to_display[0], to_display[1]))
        else:
            self.replace_quick_view_signal.emit((to_display[0], to_display[1]))
    
    def onDoubleClickedInputImage(self, item):
        selected_file = item.data(Qt.UserRole)
        selected_image_path, selected_mask_path = self.get_selected_input_instance(selected_file)

        self.open_image_signal.emit(selected_image_path)

        if selected_mask_path is not None:
            selected_mask = imageio.imread(selected_mask_path)
            self.show_segmentation_signal.emit((selected_mask, None, None))

    # Should send a signal to the parent to dispose the widget
    def onClickedBatchPredictCancel(self):
        #self.input_images_list_widget.clear()
        #self.batch_predict_widget.setVisible(False)
        #self.activate_batch_predict_btn.setEnabled(True)
        self.hide_batch_predict_widget.emit()

    @thread_worker
    def run_segmentation_batch(self, image_paths, cellsize_):
        image_paths = list(image_paths)
        for image_num, image_path in enumerate(image_paths):
            image = imageio.imread(image_path)
            mask, class_id, scores = self.nucleaizer_instance.executeSegmentation(image, cellsize_)
            
            progress = int(((image_num+1)/len(image_paths))*100)
            if mask is not None:
                save_filename = '%s.%s' % (image_path.stem, self.MASK_EXT)
                save_path = Path(self.selected_out_dir) / save_filename
                imageio.imwrite(save_path, mask)
                yield image_path.stem, progress

    def onReturnedBatchSegmentation(self):
        self.reset_status_signal.emit()
        print('Batch prediction done.')

    def onYieldedBatchSegmentation(self, stuff):
        image_name, progress = stuff
        print('Segmentation result available:', image_name)
        image_path = Path(self.selected_input_dir)/('%s.%s' % (image_name, self.IMAGE_EXT))
        mask_path = Path(self.selected_out_dir)/('%s.%s' % (image_name, self.MASK_EXT))
        
        self.update_progress_bar_signal.emit(progress)
        
        self.replace_quick_view_signal.emit((image_path, mask_path))
        self.batch_mark_completed_files()

    def get_selected_input_instance(self, selected_image_path):
        selected_mask_path = None
        if self.selected_out_dir is not None:
            selected_mask_path_ = Path(self.selected_out_dir)/('%s.%s' % (selected_image_path.stem, self.MASK_EXT))

            if selected_mask_path_.exists():
                selected_mask_path = selected_mask_path_
        
        return selected_image_path, selected_mask_path

class MeanSizeWidget(QWidget):
    
    def __init__(self, parent=None):
        super(MeanSizeWidget, self).__init__(parent)

        mean_size_layout = QHBoxLayout()
        self.setLayout(mean_size_layout)
        self.resize_widget = QCheckBox("&Expected cell size:", self)
        self.resize_widget.setChecked(True)
        self.resize_widget.stateChanged.connect(self.resize_checkbox_state_changed)
        self.cellsize = QLineEdit("20")
        mean_size_layout.addWidget(self.resize_widget)
        mean_size_layout.addWidget(self.cellsize)

    def resize_checkbox_state_changed(self, state):
        self.cellsize.setEnabled(state == Qt.Checked)

    def get_cellsize(self):
        return int(self.cellsize.text())
    
    def get_cellsize_enabled(self):
        return self.resize_widget.isChecked()

class PredictionUtils():
    def __init__(self):
        super(PredictionUtils, self).__init__()

class PredictionWidget(QGroupBox):
    
    IMAGE_PREDITC_EXTENSIONS = ['.tif', '.png', '.jpg']

    update_progress_bar_signal = Signal(int)
    reset_status_signal = Signal()
    replace_quick_view_signal = Signal(tuple)
    show_segmentation_signal = Signal(tuple)
    set_task_started_status_signal = Signal(str)
    open_image_signal = Signal(Path)

    def __init__(self, nucleaizer_env, parent=None):
        super(PredictionWidget, self).__init__(parent)
        self.parent = parent
        self.nucleaizer_env = nucleaizer_env
        self.nucleaizer_instance = None
        self.last_message_box = None

        inference_gb_layout = QVBoxLayout()
        
        self.model_selector = ModelSelectorWidget(nucleaizer_home_path=self.nucleaizer_env.get_nucleaizer_home_path())
        self.model_selector.download_progress_signal.connect(self.download_progress)
        inference_gb_layout.addWidget(self.model_selector)
        
        self.mean_size_widget = MeanSizeWidget()
        inference_gb_layout.addWidget(self.mean_size_widget)

        self.setLayout(inference_gb_layout)

        #from nucleaizer_backend.mrcnn_interface.predict import MaskRCNNSegmentation
        #self.nucleaizer_instance = MaskRCNNSegmentation()

        self.batch_predict_widget = BatchPredictionWidget(self.nucleaizer_instance, self)
        self.batch_predict_widget.hide_batch_predict_widget.connect(self.hide_batch_predict_widget)
        self.batch_predict_widget.setVisible(False)
        self.batch_predict_widget.replace_quick_view_signal.connect(self.replace_quick_view)
        self.batch_predict_widget.reset_status_signal.connect(self.reset_status)
        self.batch_predict_widget.show_segmentation_signal.connect(self.show_segmentation)
        self.batch_predict_widget.set_task_started_status_signal.connect(self.set_task_started_status)
        self.batch_predict_widget.open_image_signal.connect(self.open_image)

        self.activate_batch_predict_btn = QPushButton("Batch prediction ...")
        self.activate_batch_predict_btn.clicked.connect(self.show_batch_predict_widget)
        inference_gb_layout.addWidget(self.activate_batch_predict_btn)

        inference_gb_layout.addWidget(self.batch_predict_widget)

        self.btn_predict = QPushButton("P&redict curent image")
        self.btn_predict.clicked.connect(self.onClickPredict)
        inference_gb_layout.addWidget(self.btn_predict)

        self.model_selector.model_selected_signal.connect(self.initialize_model)

    def update_nucleaizer_instance(self, nucleaizer_instance):
        self.nucleaizer_instance = nucleaizer_instance
        self.batch_predict_widget.nucleaizer_instance = nucleaizer_instance

    @Slot(int)
    def download_progress(self, progress):
        self.update_progress_bar_signal.emit(progress)

    @Slot(Path)
    def open_image(self, path: Path):
        self.open_image_signal.emit(path)

    @Slot(tuple)
    def replace_quick_view(self, stuff):
        self.replace_quick_view_signal.emit(stuff)

    @Slot(str)
    def set_task_started_status(self, message):
        self.set_task_started_status_signal.emit(message)

    @Slot()
    def reset_status(self):
        self.reset_status_signal.emit()

    @Slot(tuple)
    def show_segmentation(self, stuff):
        self.show_segmentation_signal.emit(stuff)
    
    def show_batch_predict_widget(self):
        self.batch_predict_widget.setVisible(True)
        self.activate_batch_predict_btn.setDisabled(True)

    @Slot()
    def hide_batch_predict_widget(self):
        self.batch_predict_widget.setVisible(False)
        self.activate_batch_predict_btn.setDisabled(False)

    def onClickPredict(self):
        #if self.model_selector.dial.selected_model is None:
        if self.nucleaizer_instance is None:
            self.last_message_box = QMessageBox()
            self.last_message_box.setText("Select a model first!")
            self.last_message_box.exec()
            return            

        if self.parent.viewer.layers.selection.active is not None:
            image = self.parent.viewer.layers.selection.active.data
        else:
            self.last_message_box = QMessageBox()
            self.last_message_box.setText("Select a layer that contains an image first!")
            self.last_message_box.exec()
            return
        
        if not hasattr(image, 'ndim') or image.ndim != 3:
            self.last_message_box = QMessageBox()
            self.last_message_box.setText("Only a color image can be processed currently!")
            self.last_message_box.exec()
            print('Ndim < 3!')
            return
        
        '''
        init_success = self.initialize_model()

        if not init_success:
            msg = QMessageBox()
            msg.setText("Can't initialize the model!")
            msg.exec()
        '''

        cellsize_ = None
        if self.mean_size_widget.get_cellsize_enabled():
            cellsize_ = self.mean_size_widget.get_cellsize()
        
        #self.freeze()
        self.set_task_started_status_signal.emit('Predicting current image...')
        worker = self.run_segmentation(image, cellsize_)
        worker.returned.connect(self.onReturnedPredictSingleImage)
        worker.start()

    @thread_worker
    def run_segmentation(self, image, cellsize_):
        result = self.nucleaizer_instance.executeSegmentation(image, cellsize_)
        return result

    def initialize_model(self, selected_model):
        self.set_task_started_status_signal.emit('Initializing model...')
        self.update_progress_bar_signal.emit(50)

        #selected_model = self.model_selector.dial.selected_model
        selected_model_ = copy.copy(selected_model)

        model_meta = selected_model_.get_meta()

        from nucleaizer_backend.mrcnn_interface.predict import MaskRCNNSegmentation
        selected_model_path = selected_model_.access_resource(model_meta['model_filename'])
        nucleaizer_instance = MaskRCNNSegmentation.get_instance(selected_model_path, model_meta)
        
        self.update_nucleaizer_instance(nucleaizer_instance)
        self.reset_status_signal.emit()

    def onReturnedPredictSingleImage(self, returned_stuff):
        self.reset_status_signal.emit()
        self.show_segmentation(returned_stuff)