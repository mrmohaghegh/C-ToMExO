#! /usr/bin/env python3

# Builds the json dicts to be loaded by ctomexo

import os
import numpy as np
import pandas as pd
import json

constant1 = 1000 # the gene will be excluded if per tumor weight of clones having it is less than 1/constant1

cancer_type_list = ['LUSC', 'LUAD']
#cancer_type_list = ['LUSC', 'LUAD', 'PAAD']

for cancer_type in cancer_type_list:
    weights_dir = 'data/tracerx/raw/%s/weights'%(cancer_type)
    #weights_dir = 'data/gdac_trees/raw/%s/weights'%(cancer_type)
    clones_dir = 'data/tracerx/raw/%s/clones'%(cancer_type)
    #clones_dir = 'data/gdac_trees/raw/%s/clones'%(cancer_type)
    output_file = 'data/tracerx/%s.json'%cancer_type
    #output_file = 'data/gdac_trees/%s.json'%cancer_type

    patient_ids = [item.split('.')[0] for item in os.listdir(clones_dir) if item.endswith('.csv')]
    patient_ids.sort()

    dataset_dic = {}
    weights_dic = {}
    for patient_id in patient_ids:
        dataset_dic[patient_id] = np.array(pd.read_csv(os.path.join(clones_dir, patient_id)+'.csv'), dtype=int)
        weights_dic[patient_id] = np.array(pd.read_csv(os.path.join(weights_dir, patient_id)+'.csv'), dtype=float).reshape(-1)
    gene_names = list(pd.read_csv(os.path.join(clones_dir, patient_ids[0])+'.csv').columns)
    # removing the clones with weight equal to zero    
    for key in weights_dic.keys():
        if np.isnan(weights_dic[key][0]):
            weights_dic[key] = np.array([1.])
        elif 0 in weights_dic[key]:
            X = []
            Y = []
            for _idx, _v in enumerate(weights_dic[key]):
                if _v > 0:
                    X.append(_v)
                    Y.append(dataset_dic[key][_idx])
            X = np.array(X)
            Y = np.array(Y)
            weights_dic[key] = X
            dataset_dic[key] = Y
    # removing the genes with very few mutations
    dataset = np.concatenate([dataset_dic[key] for key in dataset_dic.keys()])
    frames = [weights_dic[key] for key in weights_dic.keys()]
    weights = np.concatenate(frames, axis=0)
    n_muts = []
    for gene in range(dataset.shape[1]):
        n_muts.append(np.sum(weights[dataset[:, gene] == 1]))
    n_muts = np.array(n_muts)
    threshold = len(patient_ids)/constant1
    genes_to_keep = n_muts > threshold
    for patient_id in patient_ids:
        dataset_dic[patient_id] = dataset_dic[patient_id][:, genes_to_keep]
    gene_names = [item for i, item in enumerate(gene_names) if genes_to_keep[i]]
    json_dataset = {
        'clones': {key:dataset_dic[key].astype(int).tolist() for key in dataset_dic.keys()},
        'weights': {key:weights_dic[key].reshape(-1).tolist() for key in weights_dic.keys()},
        'gene_names': gene_names
        }
    with open(output_file, 'w') as f:
        f.write(json.dumps(json_dataset, indent=2))
    
    # For loading:
    # with open(output_file, 'r') as f:
    #    loaded_json_dataset = json.load(f)