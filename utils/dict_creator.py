import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

def create_mappings(triples):
    """
      :param

      triples: the training triple in rdf format

      :return entity to id dictionary, relation to id dictionary:
      """
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())
    entity_to_id = {value: key for key, value in enumerate(entities)}
    rel_to_id = {value: key for key, value in enumerate(relations)}
    return entity_to_id, rel_to_id

def create_mapped_triples(triples, entity_to_id=None, rel_to_id=None):
    """
    :param

    triples: the training triple in rdf format
    entity_to_id: the dictionary of entity
    rel_to_id: the dictionary of relation to id

    :return mapped triples of ids, entity to id dictionary, relation to id dictionary

    """
    if entity_to_id is None or rel_to_id is None:
        entity_to_id, rel_to_id = create_mappings(triples)
    subject_column = np.vectorize(entity_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(triples[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    triples_of_ids = np.unique(ar=triples_of_ids, axis=0)

    return triples_of_ids, entity_to_id, rel_to_id

def write_dic(path,d):
    """
    :param

    path: path where to be saved
    d: dictionary to be written in txt

    """
    f=open (path,"w")
    keys=d.keys()
    for k in keys:
        print(str(k)+'\t'+str(d[k]))
        f.write(str(k)+'\t'+str(d[k]))
        f.write("\n")
    f.close()

def write_to_txt_file(path, data):
    """
    :param

    path: path where to be saved
    data: triples to be written in txt


    """
    f = open(path, "w")
    for i in range(data.shape[0]):
        line = ''
        for j in range(data.shape[1]):
            if(j==0):
                line = str(data[i][j])
            else:
                line = line + '\t' + str(data[i][j])
        f.write(line)
        f.write("\n")
        print(line)
    f.close()

def create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[0], entity_to_id[1]))
    rel_to_id = dict(zip(rel_to_id[0], rel_to_id[1]))

    for i in range(len(mapped_pos_train_tripels)):
        sub.append(entity_to_id[mapped_pos_train_tripels[0][i]])
        rel.append(rel_to_id[mapped_pos_train_tripels[1][i]])
        obj.append(entity_to_id[mapped_pos_train_tripels[2][i]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

def original_id_to_triples(triples, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    int_flag_sub = False
    triples = np.array(triples)
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[1], entity_to_id[0]))
    rel_to_id = dict(zip(rel_to_id[1], rel_to_id[0]))

    if (set(map(type, entity_to_id)) == {int}):
        int_flag_sub = True
    else:
        int_flag_sub = False

    if int_flag_sub == True:
        for i in range(len(triples)):
            sub.append(entity_to_id[int(triples[i][0])])
            obj.append(entity_to_id[int(triples[i][2])])
            rel.append(rel_to_id[triples[i][1]])
    else:
        for i in range(len(triples)):
            sub.append(entity_to_id[(triples[i][0])])
            obj.append(entity_to_id[(triples[i][2])])
            rel.append(rel_to_id[triples[i][1]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)



def original_id_to_triples_str(triples, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    triples = np.array(triples)
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[1], entity_to_id[0]))
    #print(entity_to_id)
    #exit()
    rel_to_id = dict(zip(rel_to_id[1], rel_to_id[0]))

    for i in range(len(triples)):
        sub.append(entity_to_id[triples[i][0]])
        rel.append(rel_to_id[triples[i][1]])
        obj.append(entity_to_id[triples[i][2]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

def mapped_id_to_original_triples(triples, entity_to_id, rel_to_id):
    """
    :param

    mapped_pos_train_tripels: the mapped ids of triples
    entity_to_id: Dictionary for entities
    rel_to_id: Dictionary for relation dictionary

    :returns

    the original triples: the original triples

    """
    triples = np.array(triples)
    sub = []
    rel = []
    obj = []
    entity_to_id = dict(zip(entity_to_id[0], entity_to_id[1]))
    rel_to_id = dict(zip(rel_to_id[0], rel_to_id[1]))

    for i in range(len(triples)):
        sub.append(entity_to_id[triples[i][0]])
        rel.append(rel_to_id[triples[i][1]])
        obj.append(entity_to_id[triples[i][2]])

    original_triples = np.c_[sub, rel, obj]
    return pd.DataFrame(original_triples)

# def swap_columns(df, c1, c2):
#     df['temp'] = df[c1]
#     df[c1] = df[c2]
#     df[c2] = df['temp']
#     df.drop(columns=['temp'], inplace=True)



data_dir = '/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/fb15k-237/mapped'
save_data_dir = 'mapped'
save_data_dir = os.path.join(data_dir,save_data_dir)
train_data_dir = os.path.join(data_dir, 'train.txt')
test_data_dir = os.path.join(data_dir, 'test.txt')
valid_data_dir = os.path.join(data_dir, 'valid.txt')

# train_triples = np.loadtxt(fname=train_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
# test_triples = np.loadtxt(fname=test_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')
# validation_triples = np.loadtxt(fname=valid_data_dir, dtype=str, comments='@Comment@ Subject Predicate Object')

train_triples = pd.read_table(train_data_dir, header=None, sep='\t')
test_triples = pd.read_table(test_data_dir, header=None, sep='\t')
validation_triples = pd.read_table(valid_data_dir, header=None, sep= '\t')


################################################################################################
#if one does not have the dictionaries then use the next block of code
#If the dataset is not ordered as subject, predicate, object and one wants to swap columns

# data_train = data_train[['subject','predicate','object']]
# data_test = data_test[['subject','predicate','object']]
# data_valid = data_valid[['subject','predicate','object']]

frames = [train_triples, test_triples, validation_triples]
result = pd.concat(frames)
result_in_array = np.array(result)

result_in_array = pd.DataFrame(result_in_array)
all_entities = list(result_in_array[0].unique()) + list(result_in_array[2].unique())
all_entities = np.unique(all_entities)
all_relations =list(result_in_array[1].unique())

pd.DataFrame(all_entities).to_csv('/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/fb15k-237/mapped/entities.dict', header=None, index=True, sep='\t')
pd.DataFrame(all_relations).to_csv('/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/fb15k-237/mapped/relations.dict', header=None, index=True, sep='\t')