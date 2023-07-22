import csv
import json

import pandas as pd
def csv_to_pandas(file_path, format):
    """This script takes a csv file, and makes a pandas out of it, but structuring it in the way specified by
    'format' """

    subjects = get_csv(file_path)

    data = []
    for subject in subjects:
        for category in format:
            structure = {}
            for el in category['static']:
                structure[el] = category['static'][el]
            for el in category['dynamic']:
                structure[el] = subject[category['dynamic'][el]]
            data.append(structure)
    return pd.DataFrame(data)
def get_csv(path):
    """This function loads and returns the contents of a csv file"""
    with open(path) as f:
        a = [{k: v for k, v in row.items()}
             for row in csv.DictReader(f, skipinitialspace=True)]
    return a
def get_json(path):
    """This function loads and returns the contents of a json file"""
    json_file = open(path)
    json_file = json_file.read()
    return json.loads(json_file)
