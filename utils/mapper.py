import pandas as pd
import numpy as np

def create_dict(data, key, value):
    dictionary = dict(zip(data[key], data[value]))
    return dictionary

def mapper(data, map_dict, cols = None):
    for col in cols:
        data[col] = data[col].map(map_dict)
    return data

train_data = pd.read_table('./data/wn18/train.txt', header = None)
test_data = pd.read_table('./data/wn18/test.txt', header = None)
valid_data = pd.read_table('./data/wn18/valid.txt', header=None)

train_data = train_data[[0,1,2]]
test_data = test_data[[0,1,2]]
valid_data = valid_data[[0,1,2]]

entity_dictionary = pd.read_table('./data/wn18/wn18_symbol_to_text.csv', header = None, sep='\t')
entity_dictionary_dictionary_map = create_dict(entity_dictionary, 0, 1)

train_data_mapped = mapper(train_data, entity_dictionary_dictionary_map, [0,2])
test_data_mapped = mapper(test_data, entity_dictionary_dictionary_map, [0,2])
valid_data_mapped = mapper(valid_data, entity_dictionary_dictionary_map, [0,2])

train_data_mapped.to_csv('./data/wn18/mapped/train.txt', header=None, index = False, sep='\t')
test_data_mapped.to_csv('./data/wn18/mapped/test.txt', header=None, index = False, sep='\t')
valid_data_mapped.to_csv('./data/wn18/mapped/valid.txt', header=None, index = False, sep='\t')