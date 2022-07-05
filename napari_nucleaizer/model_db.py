import json
from pathlib import Path

def get_models(models_dir):
    fh = open(models_dir/'models.json')
    js = json.load(fh)
    fh.close()
    return js

def main():

    models_dir = Path('/home/ervin/.nucleaizer/models')

    with open(models_dir / 'models.json', 'r') as f:
        js = json.load(f)

        for m in js['models']:
            samples_fname = m["samples"][0] if len(m["samples"]) > 0 else None
            print('---', m['id'], ':', m['name'], '-->', m['model_filename'], '---', samples_fname)

if __name__ == '__main__':
    main()