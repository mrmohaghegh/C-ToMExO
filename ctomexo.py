#! /usr/bin/env python3

# --------------------------------------------------------------------------- #
# Example:
# python ctomexo.py -i data/tracerx/LUAD.json -o result/tracerx/LUAD
#  --n_chains 3 --n_mixing 5 --n_samples 100 --pp 5
# --------------------------------------------------------------------------- #

import argparse
import numpy as np
import pandas as pd
import os
import time
from main import OncoTree
from multiprocessing import Pool
import json
from util import save_result, Geweke, Gelman_Rubin

# Hardcoded parameters:
single_error = True
error_estimation = False
collapse_interval = 10000
guided_move_prob = 0.5
thinning_interval = 10
include_linear_initializations = False
p_moves = {
    'hmerge': 1,
    'hsplit': 1,
    'vmerge': 1,
    'vsplit': 1,
    'swap': 1,
    'spr': 5,
    'gt': 5
}
factor = 1.0/sum(p_moves.values())
for k in p_moves:
    p_moves[k] = p_moves[k]*factor


def initialize(dataset_dic, weights_dic, coeff):
    if coeff < 0:
        sample = OncoTree.star_from_dataset(
            dataset_dic, weights_dic,
            single_error=single_error,
            error_estimation=error_estimation
            )
    else:
        sample = OncoTree.linear_from_dataset(
            dataset_dic, weights_dic,
            coeff=coeff,
            single_error=single_error,
            error_estimation=error_estimation
            )
    p = sample.posterior(dataset_dic, weights_dic, pp=pp)
    return(sample, p)


def run_race(args):
    sample = args[0]
    seed = args[1]
    if len(args) == 3:
        current_posterior = args[2]
    else:
        current_posterior = None
    return(sample.fast_training_iteration(
        dataset_dic, weights_dic,
        n_iters=n2,
        pp=pp,
        seed=seed,
        current_posterior=current_posterior,
        p_moves=p_moves,
        collapse_interval=collapse_interval,
        guided_move_prob=guided_move_prob,
        thinning_interval=thinning_interval,
        error_estimation=error_estimation
        ))


def fit_oncotree(
        dataset_dic,
        weights_dic,
        print_running_status=False,
        include_linear_initializations=include_linear_initializations
        ):
    st = time.time()
    if include_linear_initializations:
        coeffs = [-1]
        coeffs.extend(list(np.linspace(0, 1, n0-1)))
    else:
        coeffs = [-1 for _i in range(n0)]
    random_seeds = np.arange(1, n0+1)
    best_raw_samples = []
    sample_list = []
    p_list = []
    for coeff in coeffs:
        _sample, _p = initialize(dataset_dic, weights_dic, coeff)
        sample_list.append(_sample)
        p_list.append(_p)
    overall_posterior_array = np.empty(shape=(n0, 0))
    overall_updates_array = np.empty(shape=(n0, 0))
    details_dict = {}  # Keeping per move type stats
    for _training_race in range(n1+1):
        args_list = [(sample_list[i], random_seeds[i], p_list[i]) for i in range(len(sample_list))]
        with Pool() as p:
            results = p.map(run_race, args_list)
        if print_running_status:
            print("\nMixing point %i (out of %i) \nTime spent so far: %.1f seconds"%(_training_race, n1, time.time()-st))
        race_posterior_array = np.array([result_tuple[4] for result_tuple in results])
        overall_posterior_array = np.concatenate((overall_posterior_array, race_posterior_array), axis=1)
        race_updates_array = np.array([[result_tuple[5] for result_tuple in results]]).T
        overall_updates_array = np.concatenate((overall_updates_array, race_updates_array), axis=1)
        details_dict[_training_race] = [result_tuple[6] for result_tuple in results]
        race_tensor = np.concatenate([np.expand_dims(result_tuple[7],3) for result_tuple in results], axis=3)
        if _training_race == 0:
            posterior_tensor = np.expand_dims(race_tensor, 4)
        else:
            posterior_tensor = np.concatenate([posterior_tensor, np.expand_dims(race_tensor, 4)], axis=4)
        # Initializing the next race
        sample_list = []
        p_list = []
        best_pruned_samples = [result_tuple[2].prune(dataset_dic, weights_dic) for result_tuple in results]
        best_pruned_llhs = [pruned_sample.posterior(dataset_dic, weights_dic, pp=pp) for pruned_sample in best_pruned_samples]
        best_index = np.argmax(best_pruned_llhs)
        sample_list = [best_pruned_samples[best_index] for _i in range(n0)]
        p_list = [best_pruned_llhs[best_index] for _i in range(n0)]
        best_raw_samples.append([result_tuple[2] for result_tuple in results])
    return(best_pruned_samples[best_index], overall_posterior_array, overall_updates_array, details_dict, best_raw_samples, posterior_tensor)

# --------------------------------------------------------------------------- #
# ------------------------- Parsing command line input ---------------------- #
# --------------------------------------------------------------------------- #


parser = argparse.ArgumentParser(
    description='CTOMEXO v0.1'
    )

parser.add_argument('-i', '--input', help='input json or csv file')
parser.add_argument('-o', '--output', help='output directory')
parser.add_argument('--n_chains', help='number of chains', default=10, type=int)
parser.add_argument('--n_mixing', help='number of chain mixing events', default=0, type=int)
parser.add_argument('--n_samples', help='number of samples between mixing events', default=100000, type=int)
parser.add_argument('--pp', help='prior power', default=0, type=int)

args = parser.parse_args()

n0 = args.n_chains
n1 = args.n_mixing
n2 = args.n_samples
input_file = args.input
output_folder = args.output
pp = args.pp

# --------------------------------------------------------------------------- #
# ------------------------- Running the algorithm --------------------------- #
# --------------------------------------------------------------------------- #

if input_file.endswith('.csv'):
    df_input = pd.read_csv(input_file, delimiter=',', index_col=None, comment='#')
    if df_input.iloc[0, 0] not in [0, 1]:
        # The csv does have index column
        # RELOADING
        df_input = pd.read_csv(input_file, delimiter=',', index_col=0, comment='#')
    dataset_dic = {idx: np.array(df_input.loc[idx, :]).reshape(1, -1).astype(bool) for idx in df_input.index}
    weights_dic = {idx: np.array([[1.]]) for idx in df_input.index}
    gene_names = list(df_input.columns)
    #print('CSV input successfully loaded\n')
else:
    with open(input_file, 'r') as f:
        json_dataset = json.load(f)
    dataset_dic = {
        key: np.array(json_dataset['clones'][key]).astype(bool)
        for key in json_dataset['clones'].keys()
    }
    weights_dic = {
        key: np.array(json_dataset['weights'][key]).reshape(-1, 1)
        for key in json_dataset['weights'].keys()
    }
    gene_names = list(json_dataset['gene_names'])
    #print('JSON input successfully loaded\n')
start_time = time.time()
progmo, posterior_array, updates_array, details_dict, best_raw_samples, posterior_tensor = fit_oncotree(dataset_dic, weights_dic)
spent_time = time.time()-start_time

# --------------------------------------------------------------------------- #
# ---------------------------- Post-Processing ------------------------------ #
# --------------------------------------------------------------------------- #

save_result(os.path.join(output_folder, 'progmo.pkl'), progmo)
save_result(os.path.join(output_folder, 'raw_samples.pkl'), best_raw_samples)
np.save(os.path.join(output_folder, 'scans.npy'), posterior_tensor)

df_posterior_array = pd.DataFrame(posterior_array)
df_posterior_array.to_csv(os.path.join(output_folder, 'posteriors.csv'))

report = '# Dataset %s\n' % input_file
report += '# Number of tumors: %i\n'% len(dataset_dic.keys())
report += '# Number of genes: %i\n'% len(gene_names)
report += '# Analysis folder %s\n'% output_folder
report += '# Analysis parameters: %i chains, %i mixings, %i samples between mixings\n' %(n0, n1, n2)
report += '# Analysis time: %i seconds\n'%spent_time
report += '# ----------Convergence analysis------------ #\n'
gs_vector = posterior_array[np.argmax(np.max(posterior_array[:, -n2:], axis=1)), -n2:]
(_b, _GS) = Geweke(gs_vector)
if _b:
    report += '# Best chain has converged (Geweke z score: %.3f)\n' %_GS
else:
    report += '# Best chain has NOT converged (Geweke z score: %.3f)\n' %_GS

if n0 > 1:  # if there is more than one chain
    set_of_posts = [posterior_array[_i, -n2:] for _i in range(posterior_array.shape[0])]
    (_b, _GR) = Gelman_Rubin(set_of_posts)
    if _b:
        report += '# Gelman_Rubin convergence IS achieved (GR score: %.3f)\n'%_GR
    else:
        report += '# Gelman_Rubin convergence IS NOT achieved (GR score: %.3f)\n'%_GR

report += '# ----------Result analysis------------ # \n'

star_tree = OncoTree.star_from_dataset(
    dataset_dic, weights_dic,
    single_error=single_error,
    error_estimation=error_estimation
    )


star_llh = star_tree.likelihood(dataset_dic, weights_dic)
report += '# Star tree log likelihood: %.4f\n'%star_llh
star_pri = star_tree.prior(pp)
report += '# Star tree log prior: %.4f\n'%star_pri
report += '# Star tree log posterior: %.4f\n'%(star_pri+star_llh)

best_llh = progmo.likelihood(dataset_dic, weights_dic)
report += '# Best sample log likelihood: %.4f\n'%best_llh
best_pri = progmo.prior(pp)
report += '# Best sample log prior: %.4f\n'%best_pri
report += '# Best sample log posterior: %.4f\n'%(best_pri + best_llh)
report += '# Best sample epsilon (false positive rate): %.5f\n'%(progmo.pfp)
report += '# Best sample delta (false negative rate): %.5f\n'%(progmo.pfn)
report += '# Move type specific statistics:\n'

details_file = os.path.join(output_folder, 'analysis_details.csv')
with open(details_file, 'w') as f:
    f.write(report)

pds = []
for _race_idx in range(n1+1):
    race_details_proposed = [details_dict[_race_idx][_chain_idx]['n_proposed'] for _chain_idx in range(n0)]
    pds.append(pd.DataFrame(race_details_proposed, dtype=int, index=['n_proposed(r{}-c{})'.format(_race_idx, _chain_idx) for _chain_idx in range(n0)]))
    race_details_novel = [details_dict[_race_idx][_chain_idx]['n_novel'] for _chain_idx in range(n0)]
    pds.append(pd.DataFrame(race_details_novel, dtype=int, index=['n_novel(r{}-c{})'.format(_race_idx, _chain_idx) for _chain_idx in range(n0)]))
    race_details_accepted = [details_dict[_race_idx][_chain_idx]['n_accepted'] for _chain_idx in range(n0)]
    pds.append(pd.DataFrame(race_details_accepted, dtype=int, index=['n_accepted(r{}-c{})'.format(_race_idx, _chain_idx) for _chain_idx in range(n0)]))
pd_to_save = pd.concat(pds)
pd_to_save.to_csv(details_file, mode='a')

progmo.to_dot(dataset_dic, weights_dic, gene_names, fig_file=os.path.join(output_folder, 'best.png'))
