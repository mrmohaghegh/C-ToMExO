#! /usr/bin/env python3

# --------------------------------------------------------------------------- #
# Creates the NSCLC.json input file by combining the LUAD and LUSC files
# --------------------------------------------------------------------------- #

import json
import numpy as np

file1 = 'data/tracerx/LUAD.json'
file2 = 'data/tracerx/LUSC.json'

output_file = 'data/tracerx/NSCLC.json'

with open(file1, 'r') as f:
    json_dataset_1 = json.load(f)
dataset_dic_1 = {
    key: np.array(json_dataset_1['clones'][key]).astype(bool)
    for key in json_dataset_1['clones'].keys()
}
weights_dic_1 = {
    key: np.array(json_dataset_1['weights'][key]).reshape(-1, 1)
    for key in json_dataset_1['weights'].keys()
}
gene_names_1 = list(json_dataset_1['gene_names'])


with open(file2, 'r') as f:
    json_dataset_2 = json.load(f)
dataset_dic_2 = {
    key: np.array(json_dataset_2['clones'][key]).astype(bool)
    for key in json_dataset_2['clones'].keys()
}
weights_dic_2 = {
    key: np.array(json_dataset_2['weights'][key]).reshape(-1, 1)
    for key in json_dataset_2['weights'].keys()
}
gene_names_2 = list(json_dataset_2['gene_names'])

gene_names = []
gene_names.extend(gene_names_1)
gene_names.extend(gene_names_2)
gene_names = list(set(gene_names))
N = len(gene_names)

dataset_dic = {}
weights_dic = {}

mapping_1 = {}
for i, gene in enumerate(gene_names_1):
    mapping_1[i] = gene_names.index(gene_names_1[i])
mapping_2 = {}
for i, gene in enumerate(gene_names_2):
    mapping_2[i] = gene_names.index(gene_names_2[i])

for key in dataset_dic_1.keys():
    # build a patient
    dataset_dic[key] = np.array([np.zeros(N) for clone in dataset_dic_1[key]]).astype(bool)
    for c_idx, clone in enumerate(dataset_dic_1[key]):
        for i, gene in enumerate(clone):
            if gene:
                 dataset_dic[key][c_idx][mapping_1[i]] = True
    weights_dic[key] = weights_dic_1[key]
for key in dataset_dic_2.keys():
    # build a patient
    dataset_dic[key] = np.array([np.zeros(N) for clone in dataset_dic_2[key]]).astype(bool)
    for c_idx, clone in enumerate(dataset_dic_2[key]):
        for i, gene in enumerate(clone):
            if gene:
                dataset_dic[key][c_idx][mapping_2[i]] = True
    weights_dic[key] = weights_dic_2[key]

for patient_id in [list(dataset_dic_1.keys())[3], list(dataset_dic_1.keys())[5], list(dataset_dic_1.keys())[31], list(dataset_dic_1.keys())[4]]:
    print(patient_id)
    print(weights_dic_1[patient_id].reshape(-1))
    print([','.join([gene_names_1[i] for i, x in enumerate(clone) if x]) for clone in dataset_dic_1[patient_id]])
    print(weights_dic[patient_id].reshape(-1))
    print([','.join([gene_names[i] for i, x in enumerate(clone) if x]) for clone in dataset_dic[patient_id]])

for patient_id in [list(dataset_dic_2.keys())[3], list(dataset_dic_2.keys())[5], list(dataset_dic_2.keys())[31], list(dataset_dic_2.keys())[4]]:
    print(patient_id)
    print(weights_dic_2[patient_id].reshape(-1))
    print([','.join([gene_names_2[i] for i, x in enumerate(clone) if x]) for clone in dataset_dic_2[patient_id]])
    print(weights_dic[patient_id].reshape(-1))
    print([','.join([gene_names[i] for i, x in enumerate(clone) if x]) for clone in dataset_dic[patient_id]])


print('done')

json_dataset = {
    'clones': {key:dataset_dic[key].astype(int).tolist() for key in dataset_dic.keys()},
    'weights': {key:weights_dic[key].reshape(-1).tolist() for key in weights_dic.keys()},
    'gene_names': gene_names
    }
with open(output_file, 'w') as f:
    f.write(json.dumps(json_dataset, indent=2))

