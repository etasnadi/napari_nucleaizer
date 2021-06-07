import json

from nucleaizer_backend import load_python_module

def _load_mrcnn_config_filesystem(path):
    print('Using Mask R-CNN config:', path)
    mod = load_python_module.load("nuclei_config", path)
    return mod.NucleiConfig()

def load_mrcnn_config(nucleaizer_path, model_name=None):
    '''
    Mask R-CNN configuration loading
    1. Search a config ${model_name}_nuclei_config.py in the nucleaizer dir
    for the current model if the model name is provided
    2. If not, or the config does not exist, use a default config file
    3. If the default config file does not exist, use a default one from the package.
    '''

    if model_name is not None:
        model_config_path = nucleaizer_path / ('%s_nuclei_config.py' % model_name)
        if model_config_path.exists():
            return _load_mrcnn_config_filesystem(model_config_path)
        else:
            print('--->', model_config_path, 'does not exist!')

    default_config_path = nucleaizer_path/'nuclei_config.py'
    if default_config_path.exists():
        return _load_mrcnn_config_filesystem(default_config_path)
    
    return None

def _load_config(config_path):
    with open(config_path) as config_file:
        json_config = json.load(config_file)

    return json_config

def load_config(nucleaizer_data_path, model_name=None):    
    '''
    Load configuration for the model
    1. Try to load ${model_name}_inference.json from the nucleaizer dir if exists.
    2. If not found, try to load inference.json
    '''

    if model_name is not None:
        selected_model_config = nucleaizer_data_path/('%s_inference.json' % model_name)
        if selected_model_config.exists():
            return _load_config(selected_model_config)

    print('Config does not exist for the selected model, using the default one.')
    default_config = nucleaizer_data_path/'inference.json'
    return _load_config(default_config)