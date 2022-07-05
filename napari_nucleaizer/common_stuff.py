
from nucleaizer_backend import remote

def strip_text(text, n=6):
    sl = n
    sr = n+4
    sp = ' … '
    if len(text) > sl+sr+len(sp):
        return text[:n] + sp + text[-sr:]
    else:
        return text

'''
class ModelHandler:
    def __init__(self, nucleaizer_home_path):
        super(ModelHandler, self).__init__()
        self.nucleaizer_home_path = nucleaizer_home_path

    def get_model_list(self):
        model_root = self.nucleaizer_home_path
        
        models = list(model_root.iterdir())
        local_models = set([m.stem for m in models if m.suffix == '.h5'])
        
        remote_models = set(remote.get_model_list().keys())

        print('Available LOCAL models:', local_models)
        print('Available REMOTE models:', remote_models)
        return list(local_models.union(remote_models))
'''