#! /usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import os
from scipy.special import logsumexp, softmax
import matplotlib.pyplot as plt

####### ------------------------------------------- #######
####### ----- Main functions, frequently used ----- #######
####### ------------------------------------------- #######

def save_result(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def s_softmax(log_weights, max_ratio=1000):
    the_max = np.max(log_weights)
    the_min = np.min(log_weights)
    if the_max>the_min:
        s = np.min([1, np.log(max_ratio)/(the_max-the_min)])
        weights = softmax(s*np.array(log_weights))
    else:
        weights = np.array([1/len(log_weights)]*len(log_weights))
    return(weights)

def calc_pair_mat(dataset):
    n_tumors, n_genes = dataset.shape
    MI_scores = np.zeros((n_genes, n_genes))
    MI_log_pvalues = np.zeros((n_genes, n_genes))
    ME_log_pvalues = np.zeros((n_genes, n_genes))
    for gene1_idx in range(n_genes):
        for gene2_idx in range(gene1_idx+1, n_genes):
            mutated_in_1 = dataset[:,gene1_idx]
            mutated_in_2 = dataset[:,gene2_idx]
            mutated_in_both = mutated_in_1 * mutated_in_2
            n_1 = np.sum(mutated_in_1)
            n_2 = np.sum(mutated_in_2)
            n_both = np.sum(mutated_in_both)
            if n_1 == 0 or n_2 == 0 or n_1 == n_tumors or n_2 == n_tumors:
                MI_s = 0
                mi_log_p = 0
                me_log_p = 0
            else:
                p_1_given_2 = n_both/n_2
                p_1_given_not_2 = (n_1-n_both)/(n_tumors-n_2)
                MI_1_to_2 = (p_1_given_2-p_1_given_not_2)/(p_1_given_2+p_1_given_not_2)

                p_2_given_1 = n_both/n_1
                p_2_given_not_1 = (n_2-n_both)/(n_tumors-n_1)
                MI_2_to_1 = (p_2_given_1-p_2_given_not_1)/(p_2_given_1+p_2_given_not_1)

                me_log_p_i_1 = []
                for i in range(n_both+1):
                    me_log_p_i_1.append(
                        log_nCr(n_1, i)+(i*np.log(n_2/n_tumors))+((n_1-i)*np.log(1-n_2/n_tumors))
                    )
                me_log_p_1 = logsumexp(me_log_p_i_1)

                me_log_p_i_2 = []
                for i in range(n_both+1):
                    me_log_p_i_2.append(
                        log_nCr(n_2, i)+(i*np.log(n_1/n_tumors))+((n_2-i)*np.log(1-n_1/n_tumors))
                    )
                me_log_p_2 = logsumexp(me_log_p_i_2)

                mi_log_p_i_1 = []
                for i in range(n_both, n_1+1):
                    mi_log_p_i_1.append(
                        log_nCr(n_1, i)+(i*np.log(n_2/n_tumors))+((n_1-i)*np.log(1-n_2/n_tumors))
                    )
                mi_log_p_1 = logsumexp(mi_log_p_i_1)

                mi_log_p_i_2 = []
                for i in range(n_both, n_2+1):
                    mi_log_p_i_2.append(
                        log_nCr(n_2, i)+(i*np.log(n_1/n_tumors))+((n_2-i)*np.log(1-n_1/n_tumors))
                    )
                mi_log_p_2 = logsumexp(mi_log_p_i_2)


                MI_s = np.mean([MI_1_to_2, MI_2_to_1])
                me_log_p = logsumexp([me_log_p_1, me_log_p_2])-np.log(2)
                mi_log_p = logsumexp([mi_log_p_1, mi_log_p_2])-np.log(2)

            MI_scores[gene1_idx, gene2_idx] = MI_s
            MI_scores[gene2_idx, gene1_idx] = MI_s
            MI_log_pvalues[gene1_idx, gene2_idx] = mi_log_p
            MI_log_pvalues[gene2_idx, gene1_idx] = mi_log_p
            ME_log_pvalues[gene1_idx, gene2_idx] = me_log_p
            ME_log_pvalues[gene2_idx, gene1_idx] = me_log_p
    return(MI_scores,MI_log_pvalues, ME_log_pvalues)

def log_nCr(n, m):
    result = 0
    for i in range(m):
        result = result + np.log(n-i) - np.log(m-i)
    return(result)

def perfect_ME(node, dataset):
    
    # Returns True if the node does not violate ME
    n = len(node.genes)
    ME = True
    if n>1:
        for _idx1 in range(n-1):
            for _idx2 in range(_idx1+1, n):
                if np.sum(dataset[:,node.genes[_idx1]]*dataset[:,node.genes[_idx2]])>0:
                    ME = False
    return(ME)

def perfect_PR(node, dataset):
    
    # Returns True if the node does not violate PR
    PR = True
    if not(node.is_root) and not(node.is_ps):
        if not(node.parent.is_root):
            mutated_in_node = np.sum(dataset[:,node.genes], axis=1)>0
            mutated_in_parent = np.sum(dataset[:,node.parent.genes], axis=1)>0
            mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
            if np.sum(mutated_only_in_node)>0:
                PR = False
    return(PR)

def ME_test(node, dataset):

    # Averaged over pairs of genes (X,Y):
    # Assuming n_X >= n_Y, we go over the tumors with mutated Y and calculate
    # the ratio A/B, where:
    # A:    probability of this many or FEWER mutations in X under Null hypothesis
    # B:    probability of this many or MORE mutations in X under Null hypothesis
    n = len(node.genes)
    n_tumors = dataset.shape[0]
    if n > 1:
        me_log_p_values = []
        ME_score_values = []
        for _idx1 in range(n):
            for _idx2 in range(n):
                if _idx2 != _idx1:
                    mutated_in_1 = dataset[:,node.genes[_idx1]]
                    mutated_in_2 = dataset[:,node.genes[_idx2]]
                    mutated_in_both = mutated_in_1 * mutated_in_2
                    n_1 = np.sum(mutated_in_1)
                    n_2 = np.sum(mutated_in_2)
                    n_both = np.sum(mutated_in_both)

                    if n_1 == 0 or n_2 == 0 or n_2 == n_tumors:
                        ME_1_to_2 = 0
                        log_p = 0
                    else:
                        p_1_given_2 = n_both/n_2
                        p_1_given_not_2 = (n_1-n_both)/(n_tumors-n_2)
                        ME_1_to_2 = (p_1_given_not_2-p_1_given_2)/(p_1_given_not_2+p_1_given_2)

                        log_p_i = []
                        for i in range(n_both+1):
                            log_p_i.append(
                                log_nCr(n_1, i)+(i*np.log(n_2/n_tumors))+((n_1-i)*np.log(1-n_2/n_tumors))
                            )
                        log_p = logsumexp(log_p_i)
                    ME_score_values.append(ME_1_to_2)
                    me_log_p_values.append(log_p)

        overall_ME_score = np.mean(ME_score_values)       
        ME_p = np.mean(np.exp(me_log_p_values))
        return(overall_ME_score, ME_p)
    else:
        return()

def PR_test(node, dataset):
    # the ratio A/B, where: (note that it's NOT in log-scale)
    # A:    probability of this many or FEWER mutations in the tumors with
    #       healthy parents under Null hypothesis
    # B:    probability of this many or MORE mutations in the tumors with
    #       healthy parents under Null hypothesis
    n_tumors = dataset.shape[0]
    mutated_in_node = np.sum(dataset[:,node.genes], axis=1)>0
    mutated_in_parent = np.sum(dataset[:,node.parent.genes], axis=1)>0
    mutated_in_both = mutated_in_node * mutated_in_parent
    
    n_parent = np.sum(mutated_in_parent)
    n_node = np.sum(mutated_in_node)
    n_both = np.sum(mutated_in_both)
    n_only_node = n_node - n_both
    n_only_parent = n_parent - n_both
    n_healthy = n_tumors - n_parent - n_node + n_both

    if n_node == 0 or n_parent == 0 or n_node == n_tumors or n_parent == n_tumors:
        PR_forward = 0 # No Probability Raising
        log_p_forward = 0
        PR_backward = 0
        log_p_backward = 0
    else:
        p_node_given_parent = n_both/n_parent
        p_node_given_not_parent = n_only_node/(n_tumors-n_parent)
        PR_forward = (p_node_given_parent-p_node_given_not_parent)/(p_node_given_parent+p_node_given_not_parent)

        log_p_i_forward = []
        for i in range(n_both, n_parent+1):
            log_p_i_forward.append(
                log_nCr(n_parent, i)+(i*np.log(n_node/n_tumors))+((n_parent-i)*np.log(1-n_node/n_tumors))
            )
        log_p_forward = logsumexp(log_p_i_forward)

        p_parent_given_node = n_both/n_node
        p_parent_given_not_node = n_only_parent/(n_tumors-n_node)
        PR_backward = (p_parent_given_node-p_parent_given_not_node)/(p_parent_given_node+p_parent_given_not_node)

        log_p_i_backward = []
        for i in range(n_both, n_node+1):
            log_p_i_backward.append(
                log_nCr(n_node, i)+(i*np.log(n_parent/n_tumors))+((n_node-i)*np.log(1-n_parent/n_tumors))
            )
        log_p_backward = logsumexp(log_p_i_backward)

    score_ratio = (PR_backward+0.0001)/(PR_forward+0.0001) # will prune the edge if score_ratio>1
    F_p = np.exp(log_p_forward)
    B_p = np.exp(log_p_backward)

    return(PR_forward, F_p, PR_backward, B_p, score_ratio)

def Geweke(chain, first_proportion=0.1, second_proporiton=0.5, threshold=2):
    ''' The convergence is achieved if Z score is below threshold (the standard value is 2) '''
    n_samples = len(chain)
    if n_samples < 100:
        return(False, np.inf)
    else:
        a = chain[:int(first_proportion*n_samples)]
        b = chain[int(second_proporiton*n_samples):]
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        var_a = np.var(a)
        var_b = np.var(b)
        z_score = np.abs((mean_a-mean_b)/(np.sqrt(var_a+var_b)))
        result = z_score<threshold
        return(result, z_score)

def Gelman_Rubin(set_of_chains, burn_in=0.5, threshhold=1.2):
    ''' The convergence is achieved if Potential Scale Reduction Factor is close to 1, e.g., below 1.2 or 1.1 '''
    idx = int(len(set_of_chains[0])*(1-burn_in))
    means = np.zeros(len(set_of_chains))
    variances = np.zeros(len(set_of_chains))
    N = len(set_of_chains[0])-idx
    for _idx, chain in enumerate(set_of_chains):
        means[_idx] = np.mean(chain[idx:])
        variances[_idx] = np.var(chain[idx:],ddof=1)
    overall_mean = np.mean(means)
    N = len(set_of_chains[0])-idx
    M = len(set_of_chains)
    B = (N/(M-1))*np.sum((means-overall_mean)**2)
    W = np.mean(variances)
    V_hat = ((N-1)/N)*W + ((M+1)/(M*N))*B
    PSRF = V_hat/W
    result = PSRF < threshhold
    return(result, PSRF)

####### ------------------------------------------- #######
####### ---- Functions used for postprocessing ---- #######
####### ------------------------------------------- #######

def jacc(set1, set2):
    u = len(np.union1d(set1, set2))
    if u == 0:
        return(0)
    else:
        i = len(np.intersect1d(set1, set2, assume_unique=True))
        return((u-i)/u)

def caset(progmo, idx1, idx2):
    a1 = []
    a2 = []
    for node1 in progmo.nodes:
        if idx1 in node1.genes:
            a1.extend(node1.genes)
            for node2 in node1.ancestors:
                a1.extend(node2.genes)
            break
    for node1 in progmo.nodes:
        if idx2 in node1.genes:
            a2.extend(node1.genes)
            for node2 in node1.ancestors:
                a2.extend(node2.genes)
            break      
    return(np.intersect1d(a1, a2, assume_unique=True))

def caset_distance(sample, ref_progmo, mapping=None):
    # mapping[k] = w ---> k-th gene in the sample corresponds to w-th sample in the ref_progmo
    if mapping is None:
        mapping = {i: i for i in sample.genes}
    n = len(sample.genes)
    nom = 0
    for idx1 in range(n):
        for idx2 in range(idx1+1, n):
            nom += jacc(np.array([mapping[item] for item in caset(sample, idx1, idx2)]), caset(ref_progmo, mapping[idx1], mapping[idx2]))
    return(nom/(n*(n-1)/2))

def disc(progmo, idx1, idx2):
    a1 = []
    a2 = []
    for node1 in progmo.nodes:
        if idx1 in node1.genes:
            a1.extend(node1.genes)
            for node2 in node1.ancestors:
                a1.extend(node2.genes)
            break
    for node1 in progmo.nodes:
        if idx2 in node1.genes:
            a2.extend(node1.genes)
            for node2 in node1.ancestors:
                a2.extend(node2.genes)
            break      
    return(np.setdiff1d(a1, a2, assume_unique=True))

def disc_distance(sample, ref_progmo, mapping=None):
    # mapping[k] = w ---> k-th gene in the sample corresponds to w-th sample in the ref_progmo
    if mapping is None:
        mapping = {i: i for i in sample.genes}
    n = len(sample.genes)
    nom = 0
    for idx1 in range(n):
        for idx2 in range(n):
            if idx2 != idx1:
                nom += jacc(np.array([mapping[item] for item in disc(sample, idx1, idx2)]), disc(ref_progmo, mapping[idx1], mapping[idx2]))
    return(nom/(n*(n-1)))

def plot_df(df_input, vmin=0, vmax=2, cmap='RdBu', fontsize=12, good_to_print=lambda x: True):
    # lambda x: round(x,1)!=1
    fig, axes = plt.subplots(figsize=(10,10))
    row_label = df_input.columns
    col_label = df_input.columns
    im = axes.imshow(df_input, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7)
    axes.set_xticks(np.arange(len(col_label)))
    axes.set_yticks(np.arange(len(row_label)))
    axes.set_xticklabels(col_label, fontsize=13)
    axes.set_yticklabels(row_label, fontsize=13)
    _= plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    for _i in range(len(row_label)):
            for _j in range(len(col_label)):
                if good_to_print(df_input.iloc[_i, _j]):
                    text = axes.text(_j, _i, '%.1f'%(df_input.iloc[_i, _j]),
                                   ha="center", va="center", color="k", fontsize=fontsize)
    axes.set_xticks(np.arange(-.5, len(col_label), 1), minor=True)
    axes.set_yticks(np.arange(-.5, len(col_label), 1), minor=True)
    axes.tick_params(which='minor', length=0)
    axes.grid(which='minor', color='w', linestyle='-', linewidth=2)
    return(fig, im)