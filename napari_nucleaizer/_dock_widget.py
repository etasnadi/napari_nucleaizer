"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import math
import os
from pathlib import Path
import time
import platform

import numpy as np
import imageio
import napari
from napari_plugin_engine import napari_hook_implementation
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon, QImage, QPixmap, QFont
from qtpy.QtWidgets import (QSizePolicy, QWidget, QComboBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QListWidget, QCheckBox, QFileDialog, QMessageBox, QGroupBox, QTreeWidget, QTreeWidgetItem, QListWidgetItem, QScrollArea, QFrame, QProgressBar)
from skimage.measure import label, regionprops_table

from . import image_transform
from . import config_loader
from . import misc

from nucleaizer_backend import remote

from magicgui import magicgui
from enum import Enum

class Nucleaizer(QWidget):
    DEFAULT_STATUS = 'Ready.'
    IMAGE_EXT = 'png'
    MASK_EXT = 'tiff'

    IMAGE_PREDITC_EXTENSIONS = ['.tif', '.png', '.jpg']

    PRESEGMENTATION_DONE_SIGNAL = 0
    CLUSTERING_DONE_SIGNAL = 1
    STYLE_LEARNING_DONE_SIGNAL = 2
    TRAINING_DONE_SIGNAL = 3

    clusters_subdir = 'clusters/Kmeans-correlation-Best3Cluster'
    presegment_subdir = 'presegment'
    images_subdir = 'images'
    styles_subdir = 'styleLearnInput'

    default_home_subdir = '.nucleaizer'

    project_file_name = '.project'
    default_dir = '.'

    @staticmethod
    def strip_text(text, n=6):
        sl = n
        sr = n+4
        sp = ' … '
        if len(text) > sl+sr+len(sp):
            return text[:n] + sp + text[-sr:]
        else:
            return text
    
    @staticmethod
    def mold_image(image, expected_size):
        resize_to = expected_size[0]
        image_min = min(image.shape[:2])
        resize_factor = resize_to / image_min
        image_resized = image_transform.rescale_image(image, resize_factor)
        image_resized = image_resized[:expected_size[0], :expected_size[1], :3]
        return image_resized

    @staticmethod
    def create_collage(directory, expected_size=None, progress_bar=None):
        if expected_size is None:
            expected_size = (512, 512)
        
        images_list = list(directory.iterdir())

        n = len(images_list)
        n_cols = math.ceil(math.sqrt(n))

        blocks = []
        curr_block = []
        for idx in range(n_cols**2):
            if progress_bar is not None:
                progress = int( ((idx+1)/n)*100 )
                progress_bar.setValue(progress)

            if idx % n_cols == 0 and idx > 0:
                blocks.append(curr_block)
                curr_block = []

            if idx < len(images_list):
                curr_image = imageio.imread(images_list[idx])
                if tuple(curr_image.shape[:2]) != expected_size:
                    curr_image = Nucleaizer.mold_image(curr_image, expected_size)

                curr_image = np.transpose(curr_image, (2, 0, 1))
            else:
                curr_image = np.ones((3,) + expected_size).astype(np.uint8)


            p = 10
            curr_image = np.pad(curr_image, ((0, 0), (p, p), (p, p)))

            curr_block.append(curr_image)

        blocks.append(curr_block)

        blocks_np = np.transpose(np.block(blocks), (1, 2, 0))

        return blocks_np

    def reset_status(self):
        self.progress_bar.reset()
        self.status_bar.setText(Nucleaizer.DEFAULT_STATUS)

    def set_task_started_status(self, status=None):
        self.progress_bar.setValue(0)
        if status is not None:
            self.status_bar.setText(status)
        else:
            self.status_bar.setText('Task started...')

    def __init__(self, napari_viewer):
        super().__init__()
        
        self.dataset_path = None
        
        self.init_nucleaizer_data()
        self.models_list = self.get_model_list()

        self.nucleaizer_instance = None
        self.viewer = napari_viewer

        # GUI elements

        self.project_dir = None

        '''

        Structure: Nucleaizer
            plugin container
                scroll
                    nucleaizer_widget
                        inference_gb
                            model_list
                            mean_size
                        training_gb

        '''

        self.progress_bar = QProgressBar(self)
        self.status_bar = QLabel()

        self.reset_status()

        scroll_layout = QVBoxLayout()
        self.setLayout(scroll_layout)
        self.setMinimumWidth(300)

        inference_gb = QGroupBox("Inferencce")
        inference_gb_layout = QVBoxLayout()
        inference_gb.setLayout(inference_gb_layout)

        training_gb = QGroupBox("Training")
        training_gb_layout = QVBoxLayout()
        training_gb.setLayout(training_gb_layout)

        nucleaizer_widget = QWidget()

        nucleaizer_layout = QVBoxLayout()
        nucleaizer_layout.addWidget(inference_gb)

        if platform.system() == 'Linux':
            #nucleaizer_layout.addWidget(training_gb)
            nucleaizer_layout.addWidget(QLabel("Training will be enabled soon."))

        else:
            nucleaizer_layout.addWidget(QLabel("Training will only be supported on Linux systems."))
        
        nucleaizer_widget.setLayout(nucleaizer_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        scroll.setWidget(nucleaizer_widget)

        # Inference part
        inference_gb_layout.addWidget(QLabel('Available models'))
        def create_model_list_widget():
            modelsList = QListWidget()
            modelsList.addItems(self.models_list)
            return modelsList

        self.models_list_widget = create_model_list_widget()
        inference_gb_layout.addWidget(self.models_list_widget)

        def create_mean_size_widget():
            mean_size_widget = QWidget()
            mean_size_layout = QHBoxLayout()
            mean_size_widget.setLayout(mean_size_layout)
            self.resize_widget = QCheckBox("&Expected cell size:", self)
            self.resize_widget.stateChanged.connect(self.resize_checkbox_state_changed)
            self.cellsize = QLineEdit("20")
            mean_size_layout.addWidget(self.resize_widget)
            mean_size_layout.addWidget(self.cellsize)
            return mean_size_widget
        
        mean_size_widget = create_mean_size_widget()
        inference_gb_layout.addWidget(mean_size_widget)
        
        self.btn_predict = QPushButton("P&redict curent image")
        self.btn_predict.clicked.connect(self.onClickPredict)

        # --- Batch prediction ---
        self.selected_input_dir = None
        self.selected_out_dir = None

        self.batch_predict_widget = QWidget()
        self.batch_predict_layout = QVBoxLayout()
        self.batch_predict_widget.setLayout(self.batch_predict_layout)


        self.batch_predict_btn_text = "Batch predict selected"
        self.batch_predict_start_button = QPushButton(self.batch_predict_btn_text)
        self.batch_predict_stop_button = QPushButton("Stop")
        self.input_directory_path = QLabel("Input dir")
        self.select_input_dir_button = QPushButton("Select...")
        self.output_directory_path = QLabel("Output dir")
        self.select_output_dir_button = QPushButton("Select...")

        self.input_images_list_widget = QListWidget()

        def fill_input_images_list():
            self.input_images_list_widget.clear()
            
            for file in Path(self.selected_input_dir).iterdir():
                ext = ''

                if len(file.suffixes) > 0:
                    ext = file.suffixes[-1]
 
                if ext in self.IMAGE_PREDITC_EXTENSIONS:
                    item = QListWidgetItem(self.strip_text("%s" % file.name))
                    item.setIcon(QIcon(str(file)))
                    item.setData(Qt.UserRole, file)    
                    self.input_images_list_widget.addItem(item)

        def activate_input_images_widget():
            self.select_input_dir_button.setText(self.strip_text(self.selected_input_dir, 12))
            fill_input_images_list()
            self.batch_mark_completed_files()
            self.input_images_list_widget.setVisible(True)
            self.selection_cb.setVisible(True)
            selectionChanged(self.default_selection)
            self.selection_cb.setCurrentIndex(self.default_selection)

        def onClickSelectInputDir():
            self.selected_input_dir = QFileDialog.getExistingDirectory(
                self, "Select input directory", 
                str(self.default_dir), QFileDialog.ShowDirsOnly)

            activate_input_images_widget()
            
            if self.selected_input_dir is not None and self.selected_out_dir is not None:
                self.batch_predict_start_button.setEnabled(True)

        def onClickSelectOutputDir():
            self.selected_out_dir = QFileDialog.getExistingDirectory(
                self, "Select output directory", 
                str(self.default_dir), QFileDialog.ShowDirsOnly)

            self.select_output_dir_button.setText(self.strip_text(self.selected_out_dir, 12))
            
            if self.selected_input_dir is not None:
                activate_input_images_widget()

            if self.selected_input_dir is not None and self.selected_out_dir is not None:
                self.batch_predict_start_button.setEnabled(True)

        def onClickStopBatchPredict():
            print('Stop prediction!', self.worker.quit())

        def onClickStartBatchPredict():
            init_success = self.initialize_model()

            if not init_success:
                return False
            
            print('Batch predicting...')
            cellsize_ = int(self.cellsize.text())
            
            #images_list = Path(self.selected_input_dir).iterdir()

            images_list = get_selected_file_list()

            self.set_task_started_status('Batch predicting images...')

            self.worker = self.run_segmentation_batch(images_list, cellsize_)
            self.worker.returned.connect(self.onReturnedBatchSegmentation)
            self.worker.yielded.connect(self.onYieldedBatchSegmentation)
            self.worker.start()

        def update_selection(fun):
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

        def get_selected_file_list():
            selected_items = []
            for item_idx in range(self.input_images_list_widget.count()):
                item = self.input_images_list_widget.item(item_idx)
                if item.checkState() == Qt.Checked:
                    selected_items.append(item.data(Qt.UserRole))
            return selected_items

        def selectionChanged(i):
            print('Selected row:', i)
            selection_functions = [
                lambda x: True, 
                lambda x: self.get_selected_input_instance(x)[1] is not None,
                lambda x: self.get_selected_input_instance(x)[1] is None,
                lambda x: False]
            update_selection(selection_functions[i])

        self.batch_predict_layout.addWidget(QLabel("Predict selected"))
        self.batch_predict_cancel_button = QPushButton("Cancel")
        
        self.default_selection = 2
        self.selection_cb = QComboBox()
        self.selection_cb.addItem("All")
        self.selection_cb.addItem("Prediction exists")
        self.selection_cb.addItem("Prediction does not exist")
        self.selection_cb.addItem("None")

        self.batch_predict_layout.addWidget(self.input_directory_path)
        self.batch_predict_layout.addWidget(self.select_input_dir_button)
        self.batch_predict_layout.addWidget(self.output_directory_path)
        self.batch_predict_layout.addWidget(self.select_output_dir_button)
        self.batch_predict_layout.addWidget(self.selection_cb)
        self.batch_predict_layout.addWidget(self.input_images_list_widget)
        self.batch_predict_layout.addWidget(self.batch_predict_start_button)
        self.batch_predict_layout.addWidget(self.batch_predict_stop_button)
        self.batch_predict_layout.addWidget(self.batch_predict_cancel_button)

        def onChangedInputImage(curr, prev):
            selected_file = curr.data(Qt.UserRole)

            to_display = [selected_file, None]
            if self.selected_out_dir is not None:
                to_display = self.get_selected_input_instance(selected_file)

            self.replace_quick_view(*to_display)
        
        def onDoubleClickedInputImage(item):
            selected_file = item.data(Qt.UserRole)
            selected_image_path, selected_mask_path = self.get_selected_input_instance(selected_file)

            if selected_mask_path is not None:
                show_segmentation_result(selected_image_path, selected_mask_path)
            else:
                self.viewer.open(str(selected_image_path))

        inference_gb_layout.addWidget(self.batch_predict_widget)

        def onClickedBatchPredictCancel():
            self.input_images_list_widget.clear()

            self.batch_predict_widget.setVisible(False)
            self.activate_batch_predict_btn.setEnabled(True)
    
        self.activate_batch_predict_btn = QPushButton("Batch prediction ...")
        
        def onClickedActivateBatchPredict():
            self.batch_predict_widget.setVisible(True)
            self.activate_batch_predict_btn.setEnabled(False)

        inference_gb_layout.addWidget(self.activate_batch_predict_btn)
        inference_gb_layout.addWidget(self.btn_predict)

        self.selection_cb.setVisible(False)
        self.batch_predict_widget.setVisible(False)
        self.batch_predict_start_button.setEnabled(False)
        self.input_images_list_widget.setVisible(False)
        self.input_images_list_widget.setMinimumHeight(150)

        self.activate_batch_predict_btn.clicked.connect(onClickedActivateBatchPredict)
        self.input_images_list_widget.itemDoubleClicked.connect(onDoubleClickedInputImage)
        self.input_images_list_widget.currentItemChanged.connect(onChangedInputImage)
        self.batch_predict_cancel_button.clicked.connect(onClickedBatchPredictCancel)
        self.batch_predict_start_button.clicked.connect(onClickStartBatchPredict)
        self.batch_predict_stop_button.clicked.connect(onClickStopBatchPredict)
        self.select_output_dir_button.clicked.connect(onClickSelectOutputDir)
        self.select_input_dir_button.clicked.connect(onClickSelectInputDir)
        self.selection_cb.currentIndexChanged.connect(selectionChanged)

        # === Training part ===

        self.select_prject_button = QPushButton("Open pro&ject...")
        self.new_project_button = QPushButton("&New project...")

        #self.select_prject_button.setEnabled(False)
        #self.new_project_button.setEnabled(False)

        training_gb_layout.addWidget(self.select_prject_button)
        training_gb_layout.addWidget(self.new_project_button)

        self.training_widget = QWidget()
        self.training_widget_layout = QVBoxLayout()
        self.training_widget.setLayout(self.training_widget_layout)

        training_gb_layout.addWidget(self.training_widget)

        self.training_widget.setVisible(False)

        self.select_prject_button.clicked.connect(self.onClicedkOpenProject)
        self.new_project_button.clicked.connect(self.onClickedNewProject)

        # --- Quick view widget ---
        quick_view_gb = QGroupBox("Quick view")
        quick_view_gb_layout = QVBoxLayout()
        self.quick_view = QLabel("")
        quick_view_gb_layout.addWidget(self.quick_view)
        quick_view_gb.setLayout(quick_view_gb_layout)
        
        scroll_layout.addWidget(quick_view_gb)
        scroll_layout.addWidget(scroll)
        scroll_layout.addWidget(self.progress_bar)
        scroll_layout.addWidget(self.status_bar)

        self.quick_view.setFixedHeight(200)

        # --- Select dataset)
        
        def onClickSelectDataset():
            selected_dataset_dir = QFileDialog.getExistingDirectory(self, "Select dataset directory", self.default_dir, QFileDialog.ShowDirsOnly)

            dataset_path = Path(selected_dataset_dir)

            if dataset_path.exists():
                self.dataset_path = dataset_path
                self.create_project_file()
                self.fill_test_images_widget()

        self.select_dataset_button = QPushButton("Select dataset")
        self.training_widget_layout.addWidget(self.select_dataset_button)
        
        self.select_dataset_button.clicked.connect(onClickSelectDataset)

        # --- Test images list ---
        
        self.training_widget_layout.addWidget(QLabel('Test set & presegmentation'))
        
        def show_segmentation_result(image_path, mask_path):
            if image_path is not None and image_path.exists():
                self.viewer.open(str(image_path))
            
            if mask_path is not None and mask_path.exists():
                mask = imageio.imread(str(mask_path))
                self.show_segmentation([mask, None, None])            

        def get_test_image_instance_paths(image_path):
            return image_path, self.project_dir/self.presegment_subdir/('%s.%s' % (image_path.stem, self.MASK_EXT))

        def onDoubleClickedTestImage(item):
            selected_file = item.data(Qt.UserRole)
            show_segmentation_result(*get_test_image_instance_paths(selected_file))

        def onSelectedTestImage(curr, prev):
            if curr is not None:
                selected_file = curr.data(Qt.UserRole)
                self.replace_quick_view(*get_test_image_instance_paths(selected_file))
            
        self.test_images_widget = QListWidget()

        self.test_images_widget.setMinimumHeight(200)

        self.test_images_widget.currentItemChanged.connect(onSelectedTestImage)
        self.test_images_widget.itemDoubleClicked.connect(onDoubleClickedTestImage)

        self.training_widget_layout.addWidget(self.test_images_widget)

        # --- Cluters list ---

        def get_clustered_instace_paths(clustered_image_path):
            mask_path = self.project_dir/self.presegment_subdir/('%s.%s' % (clustered_image_path.stem, self.MASK_EXT))
            return clustered_image_path, mask_path

        def onDoubleClickedCluster(item, col):
            '''
            images/
                ${image_name}.png
            
            clusters/
                Kmeans-correlation-Best3Cluster/
                    ${group_name}/
                        ${image_name}.png
            
            presegment/
                ${image_name}.tiff
            '''

            selected_file = item.data(0, Qt.UserRole)       # The path of the image
            show_segmentation_result(*get_clustered_instace_paths(selected_file))

        def onSelectedCluster(curr, prev):
            if curr is not None:
                selected_file = curr.data(0, Qt.UserRole)
                
                if selected_file.is_dir():
                    self.set_task_started_status('Creating image collage from clustered images...')
                    collage = self.create_collage(selected_file, progress_bar=self.progress_bar)
                    self.reset_status()
                    self.viewer.add_image(collage, name='Cluster/%s' % selected_file.name)
                else:
                    self.replace_quick_view(*get_clustered_instace_paths(selected_file))

        self.clusters_widget = QTreeWidget()
        self.clusters_widget.setHeaderHidden(False)
        self.clusters_widget.setMinimumHeight(200)
        self.clusters_widget.setIconSize(QSize(24, 24))
        self.clusters_widget.setColumnCount(1)
        self.clusters_widget.setHeaderLabels(['Identified clusters'])
        self.training_widget_layout.addWidget(self.clusters_widget)

        self.clusters_widget.currentItemChanged.connect(onSelectedCluster)
        self.clusters_widget.itemDoubleClicked.connect(onDoubleClickedCluster)

        # --- Style transfer ---

        def get_style_instance_paths(image_path):
            '''
            styleLearnInput/
                ${split}/
                    p2psynthetic/
                        ${group_name}/
                            grayscale/
                                ${image_name}.tiff
                    generated/
                        ${group_name}/
                            ${image_name}.png
            '''
            split, group = image_path.parents[2].name, image_path.parents[0].name
            mask_path = self.project_dir/self.styles_subdir/split/'generated'/group/'grayscale'/('%s.%s' % (image_path.stem, self.MASK_EXT))
            return image_path, mask_path

        def onDoubleClickedStyle(item, col):
            '''
            styleLearnInput/
                ${split}/
                    p2psynthetic/
                        ${group_name}/
                            grayscale/
                                ${image_name}.tiff
                    generated/
                        ${group_name}/
                            ${image_name}.png
            '''
            selected_file = item.data(0, Qt.UserRole)   # The path of the synthetic image

            show_segmentation_result(*get_style_instance_paths(selected_file))

        def onSelectedStyle(curr, prev):
            if curr is not None:
                selected_file = curr.data(0, Qt.UserRole)
                if selected_file.is_dir():
                    self.set_task_started_status('Creating image collage from artificial images...')
                    collage = self.create_collage(selected_file, progress_bar=self.progress_bar)
                    self.reset_status()
                    self.viewer.add_image(collage, name='Style/%s' % selected_file.name)
                else:
                    self.replace_quick_view(*get_style_instance_paths(selected_file))

        #self.training_widget_layout.addWidget(QLabel('Style transfer result'))
        self.style_transfer_widget = QTreeWidget()
        self.style_transfer_widget.setHeaderHidden(False)
        self.style_transfer_widget.setMinimumHeight(200)
        self.style_transfer_widget.setColumnCount(1)
        self.style_transfer_widget.setIconSize(QSize(24, 24))
        self.style_transfer_widget.setHeaderLabels(['Style transfer result'])
        self.training_widget_layout.addWidget(self.style_transfer_widget)

        self.style_transfer_widget.currentItemChanged.connect(onSelectedStyle)
        self.style_transfer_widget.itemDoubleClicked.connect(onDoubleClickedStyle)

        # -- Training options ---
        self.presegment_check = QCheckBox("Presegmentation")
        self.training_widget_layout.addWidget(self.presegment_check)

        self.clustering_check = QCheckBox("Clustering")
        self.training_widget_layout.addWidget(self.clustering_check)
        
        self.styles_check = QCheckBox("Style learning")
        self.training_widget_layout.addWidget(self.styles_check)

        self.train_check = QCheckBox("Mask R-CNN training")
        self.training_widget_layout.addWidget(self.train_check)

        self.start_training_button = QPushButton("Train")
        self.training_widget_layout.addWidget(self.start_training_button)

        def onClickedCloseTraining():
            print('Closing project...')
            self.training_widget.setVisible(False)
            self.select_prject_button.setVisible(True)
            self.new_project_button.setVisible(True)
            self.cleanup_project()

        self.close_training_button = QPushButton("Cancel")
        self.training_widget_layout.addWidget(self.close_training_button)
        
        self.start_training_button.clicked.connect(self.onClickStartTraining)
        self.close_training_button.clicked.connect(onClickedCloseTraining)

    # ....

    def create_project_file(self):
        project_file = self.project_dir / self.project_file_name
        project_object = {'dataset_path': str(self.dataset_path)}
        misc.json_save(project_file, project_object)

    def load_project_file(self):
        project_file = self.project_dir / self.project_file_name
        project_object = misc.json_load(project_file)
        self.dataset_path = Path(project_object['dataset_path'])

    def get_selected_input_instance(self, selected_image_path):
        selected_mask_path = None
        if self.selected_out_dir is not None:
            selected_mask_path_ = Path(self.selected_out_dir)/('%s.%s' % (selected_image_path.stem, self.MASK_EXT))

            if selected_mask_path_.exists():
                selected_mask_path = selected_mask_path_
        
        return selected_image_path, selected_mask_path

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

    def cleanup_project(self):
        print('Cleaning up after closing the project.')
        self.style_transfer_widget.clear()
        self.test_images_widget.clear()
        self.clusters_widget.clear()

    def get_model_list(self):
        model_root = self.nucleaizer_home_path
        
        models = list(model_root.iterdir())
        local_models = set([m.stem for m in models if m.suffix == '.h5'])
        
        remote_models = set(remote.get_model_list().keys())

        print('Available LOCAL models:', local_models)
        print('Available REMOTE models:', remote_models)
        return list(local_models.union(remote_models))

    def resize_checkbox_state_changed(self, state):
        self.cellsize.setEnabled(state == Qt.Checked)

    def replace_quick_view(self, image_path, mask_path=None):
        '''
        A convenience function to fill the quick view window using direct path names.
        '''
        if image_path.exists():
            image = imageio.imread(image_path)
        
        mask = None
        if mask_path is not None:
            if mask_path.exists():
                mask = imageio.imread(mask_path)
        
        self.replace_quick_view_image(image, mask)

    def replace_quick_view_image(self, image, mask=None):
        '''
        Replace the quick view window with an image.
        If a mask is provided too, then draw it to the image as well.
        '''

        # Compute resize factor
        w = image.shape[0]
        target_width = 200
        scale_factor = target_width/w

        # Resize image
        image = image[..., :3]
        image = image_transform.rescale_image(image, scale_factor)
        
        # Resize mask and draw the contours on the image
        if mask is not None:
            mask = image_transform.rescale_mask(mask, scale_factor)
            image = image_transform.draw_contours(image, mask)

        rh, rw = image.shape[:2]
        image_qt = QImage(image.copy().data, rw, rh, 3*rw, QImage.Format_RGB888)

        self.quick_view.setPixmap(QPixmap.fromImage(image_qt))

    def read_project(self):
        print('Reading project...')
        self.set_task_started_status('Loading test images... (1/3)')
        self.fill_test_images_widget()
        
        self.set_task_started_status('Loading clusters... (2/3)')
        self.fill_clusters_widget()
        
        self.set_task_started_status('Loading styles... (3/3)')
        self.fill_style_transfer_widget()

        self.training_widget.setVisible(True)
        self.select_prject_button.setVisible(False)
        self.new_project_button.setVisible(False)

        self.reset_status()

    def fill_clusters_widget(self):
        print('Loading clusters to widget')

        self.clusters_widget.clear()
        clusters_dir = self.project_dir/self.clusters_subdir
        if clusters_dir.exists():
            items = [m for m in list(clusters_dir.iterdir()) if m.name[:len('group')] == 'group']
            
            items_list = sorted(items)
            for item_idx, item in enumerate(items_list):
                progress = int((item_idx+1)/len(items_list)*100)
                self.progress_bar.setValue(progress)

                widget_item = QTreeWidgetItem(self.clusters_widget)
                widget_item.setText(0, item.name)
                widget_item.setData(0, Qt.UserRole, item)

                for subitem in sorted(item.iterdir()):
                    widget_subitem = QTreeWidgetItem(widget_item)
                    ico = QIcon(str(subitem))
                    
                    widget_subitem.setText(0, self.strip_text(subitem.name))
                    widget_subitem.setData(0, Qt.UserRole, subitem)
                    widget_subitem.setIcon(0, ico)

    def fill_test_images_widget(self):
        print('Loading test images to widget!')
        
        if self.dataset_path is None:
            return

        self.test_images_widget.clear()

        presegment_dir = self.dataset_path / 'test'

        if not presegment_dir.exists():
            return

        images_list = [im for im in sorted(presegment_dir.iterdir()) if im.name.endswith(self.IMAGE_EXT)]
        
        for image_idx, image in enumerate(images_list):
            
            progress = int((image_idx+1)/len(images_list)*100)
            self.progress_bar.setValue(progress)

            widget_item = QListWidgetItem()
            widget_item.setText(self.strip_text(image.name))
            widget_item.setIcon(QIcon(str(image)))
            widget_item.setData(Qt.UserRole, image)
            self.test_images_widget.addItem(widget_item)

    def fill_style_transfer_widget(self):
        print('Loading artificial images to widget!')
        
        self.style_transfer_widget.clear()

        splits_dir = self.project_dir/self.styles_subdir
        if splits_dir.exists():
            splits_list = sorted(splits_dir.iterdir())
            for split_idx, split in enumerate(splits_list):
                styles_dir = self.project_dir/self.styles_subdir/split.name/'p2psynthetic'

                if styles_dir.exists():
                    styles_list = sorted(styles_dir.iterdir())
                    for item_idx, item in enumerate(styles_list):

                        progress_interval = 1/len(splits_list)
                        ready = progress_interval * split_idx

                        progress = int(ready*100 + ((item_idx+1)/len(styles_list)*100*progress_interval))
                        self.progress_bar.setValue(progress)

                        widget_item = QTreeWidgetItem(self.style_transfer_widget)
                        widget_item.setText(0, item.name)
                        widget_item.setData(0, Qt.UserRole, item)
                        for subitem in sorted(item.iterdir()):
                            ico = QIcon(str(subitem))
                            widget_subitem = QTreeWidgetItem(widget_item)
                            widget_subitem.setText(0, self.strip_text(subitem.name))
                            widget_subitem.setData(0, Qt.UserRole, subitem)
                            widget_subitem.setIcon(0, ico)

    def init_nucleaizer_data(self):
        '''
        Creates the Nucleaizer data directory where the models and configs are stored.
        The path is the NUCLEAIZER_DATA if the env variable is set, or ~/.nucleaizer if not.
        '''

        if 'NUCLEAIZER_DATA' in os.environ:
            self.nucleaizer_home_path = Path(os.environ['NUCLEAIZER_DATA'])
        else:
            self.nucleaizer_home_path = Path.home()/self.default_home_subdir

        print('Nucleazer data directory:', self.nucleaizer_home_path)

        if not self.nucleaizer_home_path.exists():
            print('Creating nucleaizer directory.')
            self.nucleaizer_home_path.mkdir()

    ### Prediction
    def init_nucleaizer(self, model_path):
        if self.nucleaizer_instance is None or model_path != self.loaded_model_path:
            from nucleaizer_backend.mrcnn_interface.segmentation_prediction import Segmentation
            
            model_name = model_path.stem

            #json_config = config_loader.load_config(self.nucleaizer_home_path, model_name)
            #mrcnn_config = config_loader.load_mrcnn_config(self.nucleaizer_data_path, model_name)
            
            self.nucleaizer_instance = Segmentation(str(model_path))
            self.loaded_model_path = model_path

    def onReturnedBatchSegmentation(self):
        self.reset_status()
        print('Batch prediction done.')

    def onYieldedBatchSegmentation(self, stuff):
        image_name, progress = stuff
        print('Segmentation result available:', image_name)
        image_path = Path(self.selected_input_dir)/('%s.%s' % (image_name, self.IMAGE_EXT))
        mask_path = Path(self.selected_out_dir)/('%s.%s' % (image_name, self.MASK_EXT))
        self.progress_bar.setValue(progress)
        self.replace_quick_view(image_path, mask_path)
        self.batch_mark_completed_files()

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

    @thread_worker
    def run_segmentation(self, image, cellsize_):
        result = self.nucleaizer_instance.executeSegmentation(image, cellsize_)
        return result

    ### Training

    def get_training_instance(self):
        from nucleaizer_backend.training import NucleaizerTraining
        trainer = NucleaizerTraining(
            inputs_dir=str(self.dataset_path),
            workflow_dir=str(self.project_dir),
            nucleaizer_dir=str(self.nucleaizer_home_path))
        return trainer

    @thread_worker
    def start_training(self):
        print('Starting training...')
        self.freeze()
        
        training_instance = self.get_training_instance()

        tasks = [self.presegment_check.isChecked(), self.clustering_check.isChecked(), self.styles_check.isChecked(), self.train_check.isChecked()]

        n_tasks_selected = len([t for t in tasks if t == True])
        n_tasks_finished = 0

        def get_progress():
            return int((n_tasks_finished / n_tasks_selected)*100)

        if self.presegment_check.isChecked():
            yield -1, 'Presegmentation...', get_progress()
            training_instance.copy_input()
            presegment_model = self.nucleaizer_home_path/'mask_rcnn_presegmentation.h5'
            presegment_config = self.nucleaizer_home_path/'presegment.json'
            training_instance.presegment(
                model_path=str(presegment_model), 
                config_path=str(presegment_config))
            training_instance.measure_cells()
            training_instance.setup_db()
            #time.sleep(1)
            n_tasks_finished += 1

            yield self.PRESEGMENTATION_DONE_SIGNAL, 'Presegmentation done.', get_progress()

        if self.clustering_check.isChecked():
            yield -1, 'Clustering...', get_progress()
            training_instance.clustering()
            #time.sleep(1)
            n_tasks_finished += 1

            yield self.CLUSTERING_DONE_SIGNAL, 'Clustering done.', get_progress()

        if self.styles_check.isChecked():
            yield -1, 'Style learning...', get_progress()
            training_instance.create_style_train()
            training_instance.style_transfer()
            #time.sleep(1)
            n_tasks_finished += 1

            yield self.STYLE_LEARNING_DONE_SIGNAL, 'Style learning done.', get_progress()

        if self.train_check.isChecked():
            yield -1, 'Training...', get_progress()
            # training_instance.create_mask_rcnn_train()
            # train_config = self.nucleaizer_home_path/'train.json'
            # initial_model = self.nucleaizer_home_path/'mask_rcnn_coco.h5'
            # training_instance.train_maskrcnn(
            #     config_path=train_config, 
            #     initial_model=initial_model)

            #time.sleep(1)
            n_tasks_finished += 1

            yield self.TRAINING_DONE_SIGNAL, 'Training done.', get_progress()

        self.unfreeze()

    def onReturnedPredictSingleImage(self, returned_stuff):
        self.reset_status()
        self.show_segmentation(returned_stuff)

    def show_segmentation(self, returned_stuff):
        label_image, class_ids, scores = returned_stuff

        if label_image is not None:

            # create the properties dictionary
            properties = regionprops_table(
                label_image, properties=('label', 'bbox', 'perimeter', 'area')
            )
            
            # properties['circularity'] = image_transform.circularity(
            #     properties['perimeter'], properties['area']
            # )

            # Only 8 labels can be handled right now, should fix!
            color_lut = [None, 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'white', 'orange']
            if class_ids is not None:
                properties['color'] = [color_lut[class_id] for class_id in class_ids]
            
            if scores is not None:
                properties['score'] = scores

            # create the bounding box rectangles
            bbox_rects = image_transform.make_bbox([properties[f'bbox-{i}'] for i in range(4)])

            # specify the display parameters for the text
            text_parameters = {
                'text': 'score: {score:.2f}' if 'score' in properties else 'Object',
                'size': 12,
                'color': 'green',
                'anchor': 'upper_left',
                'translation': [-3, 0],
            }

            # add the labels
            label_layer = self.viewer.add_labels(label_image, name='segmentation')

            if class_ids is not None:
                print('Class ids:', class_ids)

            shapes_layer = self.viewer.add_shapes(
                bbox_rects,
                face_color='transparent',
                edge_color=properties['color'] if 'color' in properties else 'green',
                properties=properties,
                text=text_parameters,
                name='bounding box',
            )

            self.unfreeze()

    def onClickedNewProject(self):
        '''
        qfd = QFileDialog(self)
        qfd.setOption(QFileDialog.ShowDirsOnly, True)
        qfd.setAcceptMode(QFileDialog.AcceptSave)
        qfd.selectFile(str(self.nucleaizer_data_path))
        qfd.exec()
        '''

        selected_project_dir = QFileDialog.getExistingDirectory(self, "Select project directory", self.default_dir, QFileDialog.ShowDirsOnly)
        print('New project file dialog result:', selected_project_dir)

        if len(selected_project_dir) > 0:
            self.project_dir = Path(selected_project_dir)
            if self.project_dir is None:
                print('No valid project folder is selected.')
                return
            
            print('Selected project path:', self.project_dir)
            self.read_project()

    def onClicedkOpenProject(self):
        selected_project_dir = QFileDialog.getExistingDirectory(self, "Select project directory", self.default_dir, QFileDialog.ShowDirsOnly)

        if len(selected_project_dir) > 0:
            selected_project_dir_path = Path(selected_project_dir)
            if selected_project_dir_path.exists():
                self.project_dir = selected_project_dir_path
                self.load_project_file()
                self.read_project()

    def freeze(self):
        #self.btn_predict.setEnabled(False)
        self.start_training_button.setEnabled(False)

    def unfreeze(self):
        self.btn_predict.setEnabled(True)
        self.start_training_button.setEnabled(True)

    def initialize_model(self):
        sel = self.models_list_widget.selectedIndexes()
        if len(sel) < 1:
            msg = QMessageBox()
            msg.setText("Select a model first!")
            msg.exec()
            return False

        selected_model_row = sel[0].row()
        selected_model_name = self.models_list[selected_model_row]

        selected_model_path = self.nucleaizer_home_path/('%s.h5' % selected_model_name)

        print('Selected model:', selected_model_path)

        if not selected_model_path.exists():
            print('Selected model DOES NOT exists, downloading...')
            self.set_task_started_status('Downloading model...')
            remote.download_model(selected_model_name, self.nucleaizer_home_path, self.progress_bar)
            self.reset_status()

        self.init_nucleaizer(selected_model_path)
        self.nucleaizer_instance.updateWeights(selected_model_path)

        return True

    def onClickPredict(self):
        init_success = self.initialize_model()

        if not init_success:
            return False
        
        if self.viewer.layers.selection.active is not None:
            image = self.viewer.layers.selection.active.data
        else:
            msg = QMessageBox()
            msg.setText("Select a layer that contains an image first!")
            msg.exec()
            return
        
        if image.ndim != 3:
            msg = QMessageBox()
            msg.setText("Only a color image can be processed currently!")
            msg.exec()
            print('Ndim < 3!')
            return
        
        cellsize_ = None
        if self.resize_widget.isChecked():
            cellsize_ = int(self.cellsize.text())
        
        self.freeze()
        self.set_task_started_status('Predicting current image...')
        worker = self.run_segmentation(image, cellsize_)
        worker.returned.connect(self.onReturnedPredictSingleImage)
        worker.start()

    def onTrainingPartiallyDone(self, stuff):
        signal, status, progress = stuff
        print('Training signal received:', stuff)

        if signal == self.CLUSTERING_DONE_SIGNAL:
            self.fill_clusters_widget()
        
        if signal == self.STYLE_LEARNING_DONE_SIGNAL:
            self.fill_style_transfer_widget()

        self.status_bar.setText(status)
        self.progress_bar.setValue(progress)

        #self.fill_clusters_widget()
        #self.fill_style_transfer_widget()

    def onTrainingDone(self):
        print('Training done.')
        self.reset_status()

    def onClickStartTraining(self):
        worker = self.start_training()
        worker.returned.connect(self.onTrainingDone)
        worker.yielded.connect(self.onTrainingPartiallyDone)
        worker.start()

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return Nucleaizer
