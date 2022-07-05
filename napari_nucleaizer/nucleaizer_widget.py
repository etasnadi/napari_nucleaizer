import os
from pathlib import Path
from typing import Tuple
from typing import Optional

import numpy as np
import imageio
import napari
from skimage.measure import regionprops_table
from qtpy.QtCore import Slot, Qt
from qtpy.QtGui import QImage, QPixmap, QShowEvent
from qtpy.QtWidgets import (QTabWidget, QCheckBox, QWidget, QVBoxLayout, QLabel, QGroupBox, QScrollArea, QProgressBar, QMenuBar, QMessageBox)

from . import image_transform
from . import common_stuff
from .prediction_widget import PredictionWidget
from .training_widget import TrainingWidget
from nucleaizer_backend.common import NucleaizerEnv, json_load, json_save

class Nucleaizer(QWidget):
    DEFAULT_STATUS = 'Ready.'
    IMAGE_EXT = 'png'
    MASK_EXT = 'tiff'

    default_home_subdir = '.nucleaizer'

    def __init__(self, napari_viewer: "napari.viewer.Viewer", parent: QWidget=None):
        super().__init__(parent)

        self.dataset_path = None
        self.nucleaizer_env = NucleaizerEnv()
        self.nucleaizer_env.init_nucleaizer_dir()

        self.nucleaizer_instance = None
        self.viewer = napari_viewer
        self.project_dir = None

        # GUI elements

        # Quick view widget
        self.quick_view = QLabel("")
        self.quick_view.setToolTip("The images selected in the plugin is displayed here in small version.")
        self.quick_view.setFixedHeight(200)
        quick_view_gb = QGroupBox("Quick view")
        quick_view_gb_layout = QVBoxLayout()
        quick_view_gb_layout.addWidget(self.quick_view)
        quick_view_gb.setLayout(quick_view_gb_layout)

        # NuceAIzer menubar
        self.menu = QMenuBar()
        menu = self.menu
        mi = menu.addMenu('New task...')
        
        # Outline the menu if first usage!
        outline = True
        settings = self.load_settings()
        if 'show_welcome' in settings and settings['show_welcome'] == False:
            outline = False
        if outline:
            menu.setStyleSheet("QMenuBar {border: 3px dashed red; margin: 5px; border-radius: 9px;}")
        
        mi.addAction('Prediction').triggered.connect(self.add_prediction_tab)
        mi.setToolTip("Adds a prediction task...")
        mi.addAction('New project...').triggered.connect(self.add_new_project_tab)
        mi.addAction('Open project...').triggered.connect(self.add_open_project_tab)

        # Nucleaizer tabs
        self.tab = QTabWidget()
        self.tab.setVisible(False)
        self.tab.setTabsClosable(True)
        self.tab.setMovable(True)
        self.tab.tabCloseRequested.connect(self.close_tab)

        # Main widget
        nucleaizer_widget = QWidget()
        nucleaizer_layout = QVBoxLayout()
        nucleaizer_layout.addWidget(menu)
        nucleaizer_layout.addWidget(self.tab)
        nucleaizer_widget.setLayout(nucleaizer_layout)
        
        # Status bar
        self.progress_bar = QProgressBar(self)
        self.status_bar = QLabel()
        self.reset_status()

        # Container widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(quick_view_gb)
        main_layout.addWidget(nucleaizer_widget)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_bar)
        self.setLayout(main_layout)
        self.setMinimumWidth(300)

    def startup_message(self):
        settings = self.load_settings()
        show_welcome = True
        if 'show_welcome' in settings and settings['show_welcome'] == False:
            show_welcome = False
        
        if show_welcome:
            self.last_message_box = QMessageBox()
            self.last_message_box.setWindowTitle("Napari nucleAIzer first launch")
            self.last_message_box.setTextFormat(Qt.RichText)
            txt = '''
            <h3>Welcome to the nucleAIzer plugin!</h3>
            <p>
            Using this GUI, you can exploit most of the functionalities of the nucleAIzer [1] algorithm:
            <ul>
                <li>You can use our pre-trained models to segment your images</li>
                <li>And you also can train your own models.</li>
            </ul>
            </p>

            <p>To get started, click on the <span style="display: block; font-weight: bold; border: 3px dashed red; padding: 3px;">"New task..."</span> button.

            <p>There are a few tooltips that appears when hovering on items of the GUI.</p>

            <p>Check out the <a href="https://napari-nucleaizer-docs.readthedocs.io/en/latest/?badge=latest">documentation</a> if you need further help (work in progress)!</p>

            <p>If you have found an issue, please report it on <a href="https://github.com/etasnadi/napari_nucleaizer-dev/issues">github</a>.</p>

            <p>If you want to contact the <i>plugin</i> developers, please write to <a href="mailto:tasnadi.ervin x brc.hu">tasnadi.ervin x brc.hu</a>.</p>

            <p>If you want to try the pipeline with all of the bells and whistles, check the <a href="https://github.com/spreka/biomagdsb">original code repository</a> (command line only).
            For technical support or collaboration inquiries please contact <a href="hollandi.reka x brc.hu">hollandi.reka x brc.hu</a></p>

            <p style="font-size: small;">[1] Please <a href="https://github.com/spreka/biomagdsb">cite</a> our paper if you found it useful in your research project!<p>
            '''
            msg = QCheckBox("Do not show this at startup.")
            msg.setChecked(True)
            self.last_message_box.setCheckBox(msg)
            self.last_message_box.setText(txt)
            self.last_message_box.exec()
            settings['show_welcome'] = not msg.checkState()
            self.save_settings(settings)

    def showEvent(self, a0: QShowEvent) -> None:
        self.startup_message()
        return super().showEvent(a0)

    def load_settings(self):
        settings = {}
        settings_file = self.nucleaizer_env.get_nucleaizer_home_path() / 'settings.json'
        if settings_file.exists():
            settings = json_load(settings_file)
        return settings

    def save_settings(self, settings):
        settings_file = self.nucleaizer_env.get_nucleaizer_home_path() / 'settings.json'
        json_save(settings_file, settings)

    def add_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def get_training_widget(self):
        widget = TrainingWidget(self.nucleaizer_env, self)
        widget.set_task_started_status_signal.connect(self.set_task_started_status)
        widget.update_progress_bar_signal.connect(self.update_progress_bar)
        widget.show_segmentation_signal.connect(self.show_segmentation)
        widget.open_image_signal.connect(self.open_image)
        widget.add_image_signal.connect(self.add_image)
        widget.replace_quick_view_signal.connect(self.replace_quick_view)
        widget.reset_status_signal.connect(self.reset_status)
        return widget

    def get_prediction_widget(self):
        widget = PredictionWidget(self.nucleaizer_env, self)
        widget.reset_status_signal.connect(self.reset_status)
        widget.replace_quick_view_signal.connect(self.replace_quick_view)
        widget.update_progress_bar_signal.connect(self.update_progress_bar)
        widget.show_segmentation_signal.connect(self.show_segmentation)
        widget.open_image_signal.connect(self.open_image)
        widget.set_task_started_status_signal.connect(self.set_task_started_status)
        return widget

    def remove_menu_style(self):
        self.menu.setStyleSheet("QMenuBar {padding: 5px; font-size: 16px; border: 1px dashed gray; border-radius: 5px; stroke-dasharray='10';}")

    def add_prediction_tab(self):
        self.remove_menu_style()
        self.tab.setVisible(True)
        idx = self.tab.addTab(self.add_scroll(self.get_prediction_widget()), 'prediction')
        self.tab.setCurrentIndex(idx)

    def add_training_tab(self):
        self.remove_menu_style()
        self.tab.setVisible(True)
        idx = self.tab.addTab(self.get_training_widget(), 'training')
        self.tab.setCurrentIndex(idx)

    def add_open_project_tab(self):
        self.remove_menu_style()
        self.tab.setVisible(True)
        training_widget = self.get_training_widget()
        selected_dir = training_widget.onClicedkOpenProject()
        if selected_dir != False:
            idx = self.tab.addTab(self.add_scroll(training_widget), 'training: %s'  % common_stuff.strip_text(selected_dir, 6))
            #idx = self.tab.addTab(training_widget, 'training: %s'  % common_stuff.strip_text(selected_dir, 6))
            self.tab.setTabToolTip(idx, selected_dir)
            self.tab.setCurrentIndex(idx)

    def add_new_project_tab(self):
        training_widget = self.get_training_widget()
        selected_dir = training_widget.onClickedNewProject()
        if selected_dir != False:
            idx = self.tab.addTab(self.add_scroll(training_widget), 'training: %s'  % common_stuff.strip_text(selected_dir, 6))
            self.tab.setTabToolTip(idx, selected_dir)
            self.tab.setCurrentIndex(idx)

    def close_tab(self, idx):
        self.tab.removeTab(idx)

    Slot()
    def reset_status(self):
        self.progress_bar.reset()
        self.status_bar.setText(Nucleaizer.DEFAULT_STATUS)

    Slot(str)
    def set_task_started_status(self, status: str=''):
        self.progress_bar.setValue(0)
        if len(status) > 0:
            self.status_bar.setText(status)
        else:
            self.status_bar.setText('Task started...')

    @Slot(int)
    def update_progress_bar(self, status: int):
        self.progress_bar.setValue(status)

    @Slot(tuple)
    def replace_quick_view(self, stuff: Tuple[Path, Optional[Path]]):
        '''
        A convenience function to fill the quick view window using direct path names.
        '''

        image_path = stuff[0]
        mask_path = stuff[1]

        if image_path.exists():
            image = imageio.imread(image_path)
        
        mask = None
        if mask_path is not None:
            if mask_path.exists():
                mask = imageio.imread(mask_path)
        
        self.replace_quick_view_data(image, mask)

    def replace_quick_view_data(self, image: Path, mask: Path=None):
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

    @Slot(Path)
    def open_image(self, image_path: Path):
        self.viewer.open(str(image_path))

    @Slot(np.ndarray, str)
    def add_image(self, image: np.ndarray, name: str):
        self.viewer.add_image(image, name=name)

    @Slot(tuple)
    def show_segmentation(self, returned_stuff):
        label_image, class_ids, scores = returned_stuff[0], returned_stuff[1], returned_stuff[2]

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