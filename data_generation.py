#! /usr/bin/env python3

# Example: onco, average_size, average_depth = create_oncotree(5, 10, 0.5, 1, pfp=0.01, pfn=0.01, single_error=True)

import numpy as np
import os
import pickle
import json
from main import OncoNode, OncoTree
from anytree import PreOrderIter
from matplotlib import pyplot as plt


def calculate_mut_rates(gen_progmo, alpha):
    new_mut_rates = [1] * gen_progmo.n_genes
    for node in PreOrderIter(gen_progmo.root):
        if not node.is_root:
            _f = node.f
            for anc_node in node.ancestors:
                _f *= anc_node.f
            prob_simplex = np.random.default_rng().dirichlet(alpha*np.ones(len(node.genes)))
            prob_simplex *= _f
            for idx, gene in enumerate(node.genes):
                new_mut_rates[gene] = prob_simplex[idx]
    gen_progmo.mut_rates = np.array(new_mut_rates)
    return(gen_progmo)


def tree_depth(gen_progmo):
    tree_depth = 0
    for node in PreOrderIter(gen_progmo.root):
        tree_depth += node.depth * len(node.genes)
    tree_depth = tree_depth / gen_progmo.n_genes
    return(tree_depth)


def create_oncotree_structure(K, N, mu):
    # Input
    # K: number of nodes, N: number of genes, mu: probability of linear expansion
    # pfp: FP probability, pfn: FN probability

    # Hardcoded parameters:
    f_low = 0.05
    f_high = 0.95

    genes = [x for x in range(N)]
    temp_mut_rates = [1]*len(genes)
    nodes = [OncoNode(genes=[], f=1)]
    for _node in range(1, K+1):
        fir_prob = float(np.random.uniform(low=f_low, high=f_high, size=1))
        if (_node == 1) | bool(np.random.binomial(1, mu)):
            nodes.append(OncoNode(genes=[genes.pop(0)], f=fir_prob, parent=nodes[-1]))
        else:
            par_idx = np.random.randint(0, len(nodes)-1)  # exclude the last node
            nodes.append(OncoNode(genes=[genes.pop(0)], f=fir_prob, parent=nodes[par_idx]))
    while len(genes) > 0:  # Add the remaining genes
        chosen_idx = np.random.randint(1, len(nodes))
        nodes[chosen_idx].genes.append(genes.pop(0))
    return(nodes)


def create_progmo(nodes, alpha, pfp=0.2, pfn=0.2, single_error=True):
    # alpha: dirichlet parameter
    gen_progmo = OncoTree(nodes, pfp=pfp, pfn=pfn, single_error=single_error)
    gen_progmo = calculate_mut_rates(gen_progmo, alpha)
    aver_tree_depth = tree_depth(gen_progmo)
    aver_node_size = np.sum([len(node.genes) for node in nodes])/(len(nodes)-1)
    return(gen_progmo, aver_node_size, aver_tree_depth)


if __name__ == '__main__':

    np.random.seed(1)

    trees_dir = 'data/synthetic/progmos/'
    os.makedirs(os.path.dirname(trees_dir), exist_ok=True)

    datasets_dir = 'data/synthetic/datasets/'
    os.makedirs(os.path.dirname(datasets_dir), exist_ok=True)

    n_progmos = 100
    e_values = [0.001, 0.01, 0.02]
    n_tumors_values = [10, 30, 100, 300]
    min_n_genes = 10
    max_n_genes = 40
    min_mu = 0
    max_mu = 1
    max_mean_genes_per_node = 4
    max_mean_gene_depth = 4
    alpha = 1

    trees_list = []
    size_list = []
    depth_list = []
    while len(trees_list) < n_progmos:
        n_genes = np.random.choice(np.arange(min_n_genes, max_n_genes))
        n_nodes = np.random.choice(np.arange(int(np.ceil(n_genes/max_mean_genes_per_node)), n_genes+1))
        mu = np.random.uniform(min_mu, max_mu)
        tmp_nodes = create_oncotree_structure(n_nodes, n_genes, mu)
        tmp_tree, tmp_size, tmp_depth = create_progmo(tmp_nodes, alpha)
        if tmp_depth < max_mean_gene_depth:
            trees_list.append(tmp_tree)
            size_list.append(tmp_size)
            depth_list.append(tmp_depth)
    for i, tree in enumerate(trees_list):
        file_path = os.path.join(trees_dir, 't%i_s%.3f_d%.3f.pkl' % (i, size_list[i], depth_list[i]))
        with open(file_path, 'wb') as f:
            pickle.dump(tree, f)
        img_path = os.path.join(trees_dir, 't%i.png' % (i))
        _ = tree.to_dot(fig_file=img_path)

    #  Saving the size vs depth plot
    fig, ax = plt.subplots()
    ax.scatter(size_list, depth_list)
    ax.grid()
    ax.set_xlabel('genes per node')
    ax.set_ylabel('gene depth')
    plt.savefig('data/synthetic/progmos/depth_vs_size.pdf')

    #  Creating the datasets
    for i, t in enumerate(trees_list):
        gene_names = ['g%i' % idx for idx in t.genes]
        for e in e_values:
            t.pfp = e
            t.pfn = e
            for n_tumors in n_tumors_values:
                output_file = os.path.join(datasets_dir, 't%i_e%.3f_n%i.json' % (i, e, n_tumors))
                dataset_dic, weights_dic = t.draw_sample(n_tumors, a=100)

                # The preprocessing used for omitting rarely mutated genes:
                dataset = np.concatenate([dataset_dic[key] for key in dataset_dic.keys()])
                frames = [weights_dic[key] for key in weights_dic.keys()]
                weights = np.concatenate(frames, axis=0)
                n_muts = []
                for gene in range(dataset.shape[1]):
                    n_muts.append(np.sum(weights[dataset[:, gene] == 1]))
                n_muts = np.array(n_muts)
                constant1 = 1000
                threshold = len(dataset_dic)/constant1
                genes_to_keep = n_muts > threshold
                for key in dataset_dic.keys():
                    dataset_dic[key] = dataset_dic[key][:, genes_to_keep]
                dataset_gene_names = [item for i, item in enumerate(gene_names) if genes_to_keep[i]]

                # Creating the json file
                json_dataset = {
                    'clones': {key: dataset_dic[key].astype(int).tolist() for key in dataset_dic.keys()},
                    'weights': {key: weights_dic[key].reshape(-1).tolist() for key in weights_dic.keys()},
                    'gene_names': dataset_gene_names
                    }
                with open(output_file, 'w') as f:
                    f.write(json.dumps(json_dataset, indent=2))
