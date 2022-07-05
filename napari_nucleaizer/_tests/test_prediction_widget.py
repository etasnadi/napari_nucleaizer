def _get_constructor_args():
    return '/home/ervin/.nucleaizer'

'''
def _test_prediction_widget(qtbot):
    nucleaizer_home = _get_constructor_args()
    from napari_nucleaizer.prediction_widget import PredictionWidget
    from nucleaizer_backend.common import NucleaizerEnv
    env = NucleaizerEnv()
    pw = PredictionWidget(env)
    qtbot.addWidget(pw)
    assert pw.nucleaizer_instance == None
'''

'''
def _test_prediction_button_no_model_selected(qtbot):
    nucleaizer_home = _get_constructor_args()
    from napari_nucleaizer.prediction_widget import PredictionWidget
    from nucleaizer_backend.common import NucleaizerEnv
    env = NucleaizerEnv()
    pw = PredictionWidget(env)
    qtbot.addWidget(pw)
    pw.show()
    qtbot.wait_for_window_shown(pw)
    
    import qtpy.QtCore
    def on_timeout():
        assert pw.last_message_box is not None
        assert pw.last_message_box.text() == "Select a model first!"
        pw.last_message_box.close()
        
    qtpy.QtCore.QTimer.singleShot(1, on_timeout)
    qtbot.mouseClick(pw.btn_predict, qtpy.QtCore.Qt.LeftButton)
'''

def test_dummy():
    def always_true():
        return True
    
    assert always_true()==True