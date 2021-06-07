import json

def json_load(file):
    data = None
    with open(file) as fp:
        data = json.load(fp)

    return data

def json_save(file, data):
    with open(file, 'w') as fp:
        json.dump(data, fp, indent=4)
