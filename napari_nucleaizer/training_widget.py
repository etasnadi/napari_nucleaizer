import math
import copy
from pathlib import Path
import shutil

import jsonpickle
import imageio
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtGui import QIcon, QFont
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget, QCheckBox, QFileDialog, QGroupBox, QTreeWidget, QTreeWidgetItem, QListWidgetItem, QScrollArea)

from nucleaizer_backend.common import json_load, json_save
from . import common_stuff
from . import image_transform
from napari_nucleaizer.model_selector_widget import ModelSelectorWidget

class TrainingWidget(QGroupBox):
    
    IMAGE_EXT = 'png'
    MASK_EXT = 'tiff'

    PRESEGMENTATION_DONE_SIGNAL = 0
    CLUSTERING_DONE_SIGNAL = 1
    STYLE_LEARNING_DONE_SIGNAL = 2
    TRAINING_DONE_SIGNAL = 3

    clusters_subdir = 'clusters/Kmeans-correlation-Best3Cluster'
    presegment_subdir = 'presegment'
    images_subdir = 'images'
    styles_subdir = 'styleLearnInput'

    project_file_name = '.project'

    #default_dir = '/home/ervin/nucleaizer_demo_folder/nucleaizer_workflow_phase2'
    #default_dir_new_project = '/home/ervin/nucleaizer_demo_folder/nucleaizer_workflow_phase2'
    
    default_dir = '.'
    default_dir_new_project = '.'

    set_task_started_status_signal = Signal(str)
    update_progress_bar_signal = Signal(int)
    show_segmentation_signal = Signal(tuple)
    open_image_signal = Signal(Path)
    add_image_signal = Signal(np.ndarray, str)
    replace_quick_view_signal = Signal(tuple)
    reset_status_signal = Signal()

    def __init__(self, nucleaizer_env, parent=None):
        # --- Select dataset)
        super(TrainingWidget, self).__init__(parent)

        # Stuff that can be saved into the project file.
        self.dataset_path = None
        self.presegment_model = None
        self.pretrained_model = None
        self.project_object = {}

        self.nucleaizer_env = nucleaizer_env

        training_gb_layout = QVBoxLayout()
        self.setLayout(training_gb_layout)

        self.training_widget = QWidget()
        self.training_widget_layout = QVBoxLayout()
        self.training_widget.setLayout(self.training_widget_layout)

        training_gb_layout.addWidget(self.training_widget)

        #self.training_widget.setVisible(False)

        self.select_dataset_button = QPushButton("Select dataset...")
        self.training_widget_layout.addWidget(QLabel("Dataset folder (with train, val & test subfolders)"))
        self.training_widget_layout.addWidget(self.select_dataset_button)
        self.select_dataset_button.clicked.connect(self.onClickSelectDataset)

        self.style_learning = QCheckBox("Full training with style transfer")
        self.style_learning.setChecked(True)
        self.style_learning.stateChanged.connect(self.style_learning_switch)

        self.model_selector_presegmentation = ModelSelectorWidget(nucleaizer_home_path=self.nucleaizer_env.get_nucleaizer_home_path(), select_model_text='Presegmentation profile: ')
        self.training_widget_layout.addWidget(self.model_selector_presegmentation)
        self.model_selector_presegmentation.model_selected_signal.connect(self.presegmentation_model_selected)

        self.training_set_label = QLabel('Training set: 0')
        self.training_set_label.setVisible(False)
        self.validation_set_label = QLabel('Validation set: 0')
        self.validation_set_label.setVisible(False)
        self.training_widget_layout.addWidget(self.training_set_label)
        self.training_widget_layout.addWidget(self.validation_set_label)

        self.test_images_label = QLabel('Test set')
        self.training_widget_layout.addWidget(self.test_images_label)
        self.test_images_widget = QListWidget()
        self.test_images_widget.setMinimumHeight(200)
        self.test_images_widget.currentItemChanged.connect(self.onSelectedTestImage)
        self.test_images_widget.itemDoubleClicked.connect(self.onDoubleClickedTestImage)
        self.training_widget_layout.addWidget(self.test_images_widget)

        self.training_widget_layout.addWidget(self.style_learning)

        def get_tree_widget(label: str):
            tree_widget = QTreeWidget()
            tree_widget.setHeaderHidden(False)
            tree_widget.setMinimumHeight(200)
            tree_widget.setIconSize(QSize(24, 24))
            tree_widget.setColumnCount(1)
            tree_widget.setHeaderLabels([label])
            return tree_widget

        self.clusters_widget = get_tree_widget('Identified clusters')
        self.training_widget_layout.addWidget(self.clusters_widget)
        self.clusters_widget.currentItemChanged.connect(self.onTreeItemSelected)
        self.clusters_widget.itemDoubleClicked.connect(self.onDoubleClickedCluster)

        self.style_transfer_widget = get_tree_widget('Style transfer result')
        self.training_widget_layout.addWidget(self.style_transfer_widget)
        self.style_transfer_widget.currentItemChanged.connect(self.onTreeItemSelected)
        self.style_transfer_widget.itemDoubleClicked.connect(self.onDoubleClickedStyle)

        self.model_selector_training = ModelSelectorWidget(nucleaizer_home_path=self.nucleaizer_env.get_nucleaizer_home_path(), select_model_text='Mask R-CNN initialization profile:')
        self.training_widget_layout.addWidget(self.model_selector_training)
        self.model_selector_training.model_selected_signal.connect(self.pretrained_model_selected)

        # -- Training options ---
        self.presegment_check = QCheckBox("Presegmentation")
        self.training_widget_layout.addWidget(self.presegment_check)

        self.clustering_check = QCheckBox("Clustering")
        self.training_widget_layout.addWidget(self.clustering_check)
        
        self.styles_check = QCheckBox("Style learning")
        self.training_widget_layout.addWidget(self.styles_check)

        self.train_check = QCheckBox("Mask R-CNN training")
        self.training_widget_layout.addWidget(self.train_check)

        self.start_training_button = QPushButton("Run pipeline")
        self.training_widget_layout.addWidget(self.start_training_button)
        
        self.start_training_button.clicked.connect(self.onClickStartTraining)

    def download_training_assets(self):
        '''
        Downloads the assets needed for training into $NUCLEAIZER_HOME:
        clustering/
        native_linux_x86-64/
        sac/
        train.json
        '''
        # self.nucleaizer_home_path
        pass

    def pretrained_model_selected(self, selected_model):
        self.pretrained_model = selected_model
        self.create_project_file()

    def presegmentation_model_selected(self, selected_model):
        self.presegment_model = selected_model
        self.create_project_file()

    def style_learning_switch(self):
        check_status = self.style_learning.isChecked()

        widgets_to_toggle = [self.model_selector_presegmentation , self.clusters_widget, self.style_transfer_widget, self.presegment_check, self.clustering_check, self.styles_check]
        
        for widget in widgets_to_toggle:
            widget.setEnabled(check_status)

        self.presegment_check.setCheckState(False)
        self.clustering_check.setCheckState(False)
        self.styles_check.setCheckState(False)

    def onClickSelectDataset(self):
        selected_dataset_dir = QFileDialog.getExistingDirectory(self, "Select dataset directory", self.default_dir, QFileDialog.ShowDirsOnly)

        if len(selected_dataset_dir) < 1:
            return

        dataset_path = Path(selected_dataset_dir)

        if dataset_path.exists():
            self.dataset_path = dataset_path
            self.select_dataset_button.setText(str(dataset_path))
            self.create_project_file()
            self.fill_test_images_widget()
        
    def show_segmentation_result(self, image_path, mask_path):
        if image_path is not None and image_path.exists():
            self.open_image_signal.emit(image_path)
        
        if mask_path is not None and mask_path.exists():
            mask = imageio.imread(str(mask_path))
            self.show_segmentation_signal.emit((mask, None, None))            

    def get_test_image_instance_paths(self, image_path):
        return image_path, self.project_dir/self.presegment_subdir/('%s.%s' % (image_path.stem, self.MASK_EXT))

    def onDoubleClickedTestImage(self, item):
        selected_file = item.data(Qt.UserRole)
        self.show_segmentation_result(*self.get_test_image_instance_paths(selected_file))

    def onSelectedTestImage(self, curr, prev):
        if curr is not None:
            selected_file = curr.data(Qt.UserRole)
            self.replace_quick_view_signal.emit(self.get_test_image_instance_paths(selected_file))

        # --- Cluters list ---

    def get_clustered_instace_paths(self, clustered_image_path):
        mask_path = self.project_dir/self.presegment_subdir/('%s.%s' % (clustered_image_path.stem, self.MASK_EXT))
        return clustered_image_path, mask_path

    def onDoubleClickedCluster(self, item, col):
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
        self.show_segmentation_result(*self.get_clustered_instace_paths(selected_file))

    def onTreeItemSelected(self, curr, prev):
        if curr is not None:
            selected_file = curr.data(0, Qt.UserRole)
            
            if selected_file.is_dir():
                self.set_task_started_status_signal.emit('Creating image collage...')
                collage = self.create_collage(selected_file, progress_bar=self.update_progress_bar_signal)
                self.reset_status_signal.emit()
                self.add_image_signal.emit(collage, 'Collage/%s' % selected_file.name)
            else:
                self.replace_quick_view_signal.emit(self.get_clustered_instace_paths(selected_file))

    # --- Style transfer ---

    def get_style_instance_paths(self, image_path):
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

    def onDoubleClickedStyle(self, item, col):
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

        self.show_segmentation_result(*self.get_style_instance_paths(selected_file))

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
                progress_bar.emit(progress)

            if idx % n_cols == 0 and idx > 0:
                blocks.append(curr_block)
                curr_block = []

            if idx < len(images_list):
                curr_image = imageio.imread(images_list[idx])
                if tuple(curr_image.shape[:2]) != expected_size:
                    curr_image = TrainingWidget.mold_image(curr_image, expected_size)

                curr_image = np.transpose(curr_image, (2, 0, 1))
            else:
                curr_image = np.ones((3,) + expected_size).astype(np.uint8)


            p = 10
            curr_image = np.pad(curr_image, ((0, 0), (p, p), (p, p)))

            curr_block.append(curr_image)

        blocks.append(curr_block)

        blocks_np = np.transpose(np.block(blocks), (1, 2, 0))

        return blocks_np

    def get_training_instance(self):
        from nucleaizer_backend.training import NucleaizerTraining
        trainer = NucleaizerTraining(
            inputs_dir=str(self.dataset_path),
            workflow_dir=str(self.project_dir),
            nucleaizer_home_path=str(self.nucleaizer_env.get_nucleaizer_home_path()))
        return trainer

    @thread_worker
    def start_training(self):
        print('Running pipeline...')
        self.freeze()
        
        training_instance = self.get_training_instance()
        training_instance.init()


        tasks = [self.presegment_check.isChecked(), self.clustering_check.isChecked(), self.styles_check.isChecked(), self.train_check.isChecked()]

        n_tasks_selected = len([t for t in tasks if t == True])
        n_tasks_finished = 0

        def get_progress():
            return int((n_tasks_finished / n_tasks_selected)*100)

        if self.presegment_check.isChecked():
            yield -1, 'Presegmentation...', get_progress()
            training_instance.copy_input()
            selected_model = self.model_selector_presegmentation.dial.selected_model
            
            selected_model = self.model_selector_presegmentation.dial.selected_model
            selected_model_ = copy.copy(selected_model)
            model_meta = selected_model_.get_meta()
            
            from nucleaizer_backend.mrcnn_interface.predict import MaskRCNNSegmentation
            selected_model_path = selected_model_.access_resource(model_meta['model_filename'], self.update_progress_bar_signal)
            nucleaizer_instance = MaskRCNNSegmentation.get_instance(selected_model_path, model_meta)

            training_instance.presegment(nucleaizer_instance)
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
            yield -1, 'Training Mask R-CNN...', get_progress()

            selected_model = self.model_selector_training.dial.selected_model
            
            selected_model = self.model_selector_training.dial.selected_model
            selected_model_ = copy.copy(selected_model)
            initial_model = selected_model_.access_resource(selected_model_.get_meta()['model_filename'], self.update_progress_bar_signal)

            if self.style_learning.isChecked():
                training_instance.create_mask_rcnn_train()
                training_instance.crop_training_set()

            training_instance.train_maskrcnn(
                initial_model=initial_model,
                use_style_augmentation=self.style_learning.isChecked())

            training_instance.deploy_model()

            n_tasks_finished += 1
            yield self.TRAINING_DONE_SIGNAL, 'Training done.', get_progress()

        self.unfreeze()

    def onClickedNewProject(self):
        '''
        qfd = QFileDialog(self)
        qfd.setOption(QFileDialog.ShowDirsOnly, True)
        qfd.setAcceptMode(QFileDialog.AcceptSave)
        qfd.selectFile(str(self.nucleaizer_data_path))
        qfd.exec()
        '''

        selected_project_dir = QFileDialog.getExistingDirectory(self, "Select project directory", self.default_dir_new_project, QFileDialog.ShowDirsOnly)
        print('New project file dialog result:', selected_project_dir)

        if len(selected_project_dir) > 0:
            self.project_dir = Path(selected_project_dir)
            if self.project_dir is None:
                print('No valid project folder is selected.')
                return False
            
            print('Selected project path:', self.project_dir)
            self.read_project()

            # Copy the template train.py from the to this folder.
            shutil.copy(self.nucleaizer_env.get_nucleaizer_home_path()/'train.py', self.project_dir/'train.py')
        else:
            return False

    def onClicedkOpenProject(self):
        selected_project_dir = QFileDialog.getExistingDirectory(self, "Select project directory", self.default_dir, QFileDialog.ShowDirsOnly)

        if len(selected_project_dir) > 0:
            selected_project_dir_path = Path(selected_project_dir)
            if selected_project_dir_path.exists():
                self.project_dir = selected_project_dir_path
                self.load_project_file()
                self.read_project()
                self.select_dataset_button.setText(common_stuff.strip_text(str(self.dataset_path), 8))
                self.select_dataset_button.setToolTip(str(self.dataset_path))
                if self.presegment_model is not None:
                    self.model_selector_presegmentation.select_model(self.presegment_model)
                if self.pretrained_model is not None:
                    self.model_selector_training.select_model(self.pretrained_model)
                return selected_project_dir
            else:
                return False
        else:
            return False

    def onTrainingPartiallyDone(self, stuff):
        signal, status, progress = stuff
        print('Training signal received:', stuff)

        if signal == self.CLUSTERING_DONE_SIGNAL:
            self.fill_clusters_widget()
        
        if signal == self.STYLE_LEARNING_DONE_SIGNAL:
            self.fill_style_transfer_widget()

        self.set_task_started_status_signal.emit(status)
        self.update_progress_bar_signal.emit(progress)

        #self.fill_clusters_widget()
        #self.fill_style_transfer_widget()

    def onTrainingDone(self):
        print('Training done.')
        self.reset_status_signal.emit()

    def onClickStartTraining(self):
        worker = self.start_training()
        worker.returned.connect(self.onTrainingDone)
        worker.yielded.connect(self.onTrainingPartiallyDone)
        worker.start()

    def freeze(self):
        self.start_training_button.setEnabled(False)

    def unfreeze(self):
        self.start_training_button.setEnabled(True)
    
    def read_project(self):
        print('Reading project...')

        self.set_task_started_status_signal.emit('Loading test images... (1/3)')
        self.fill_test_images_widget()
        
        self.set_task_started_status_signal.emit('Loading clusters... (2/3)')
        self.fill_clusters_widget()
        
        self.set_task_started_status_signal.emit('Loading styles... (3/3)')
        self.fill_style_transfer_widget()

        self.training_widget.setVisible(True)
        self.reset_status_signal.emit()

        train_size = len(list((self.dataset_path/'train'/'images').iterdir()))
        val_size = len(list((self.dataset_path/'val'/'images').iterdir()))
        self.training_set_label.setText("Training set (%d)" % train_size)
        self.validation_set_label.setText("Validation set (%d)" % val_size)
        self.training_set_label.setVisible(True)
        self.validation_set_label.setVisible(True)

    def fill_clusters_widget(self):
        print('Loading clusters to widget')

        self.clusters_widget.clear()
        clusters_dir = self.project_dir/self.clusters_subdir
        if clusters_dir.exists():
            items = [m for m in list(clusters_dir.iterdir()) if m.name[:len('group')] == 'group']
            
            items_list = sorted(items)
            for item_idx, item in enumerate(items_list):
                progress = int((item_idx+1)/len(items_list)*100)
                self.update_progress_bar_signal.emit(progress)

                widget_item = QTreeWidgetItem(self.clusters_widget)
                widget_item.setData(0, Qt.UserRole, item)

                subitems = sorted(item.iterdir())
                for subitem in subitems:
                    widget_subitem = QTreeWidgetItem(widget_item)
                    ico = QIcon(str(subitem))
                    
                    widget_subitem.setText(0, common_stuff.strip_text(subitem.name))
                    widget_subitem.setData(0, Qt.UserRole, subitem)
                    widget_subitem.setIcon(0, ico)

                widget_item.setText(0, '%s (%d)' % (item.name, len(subitems)))

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
            self.update_progress_bar_signal.emit(progress)

            widget_item = QListWidgetItem()
            widget_item.setText(common_stuff.strip_text(image.name))
            widget_item.setIcon(QIcon(str(image)))
            widget_item.setData(Qt.UserRole, image)
            self.test_images_widget.addItem(widget_item)

        self.test_images_label.setText("Test set (%d)" % len(images_list))

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
                        self.update_progress_bar_signal.emit(progress)

                        widget_item = QTreeWidgetItem(self.style_transfer_widget)
                        widget_item.setData(0, Qt.UserRole, item)
                        subitems = sorted(item.iterdir())
                        for subitem in subitems:
                            ico = QIcon(str(subitem))
                            widget_subitem = QTreeWidgetItem(widget_item)
                            widget_subitem.setText(0, common_stuff.strip_text(subitem.name))
                            widget_subitem.setData(0, Qt.UserRole, subitem)
                            widget_subitem.setIcon(0, ico)

                        widget_item.setText(0, '%s (%d)' % (item.name, len(subitems)))

    def cleanup_project(self):
        print('Cleaning up after closing the project.')
        self.style_transfer_widget.clear()
        self.test_images_widget.clear()
        self.clusters_widget.clear()

    def create_project_file(self):
        project_file = self.project_dir / self.project_file_name
        project_object = {
            'dataset_path': str(self.dataset_path), 
            'presegment_model': jsonpickle.encode(self.presegment_model),
            'pretrained_model': jsonpickle.encode(self.pretrained_model)
        }
        json_save(project_file, project_object)

    def load_project_file(self):
        project_file = self.project_dir / self.project_file_name
        project_object = json_load(project_file)
        self.dataset_path = Path(project_object['dataset_path'])
        if 'presegment_model' in project_object:
            self.presegment_model = jsonpickle.decode(project_object['presegment_model'])
        
        if 'pretrained_model' in project_object:
            self.pretrained_model = jsonpickle.decode(project_object['pretrained_model'])