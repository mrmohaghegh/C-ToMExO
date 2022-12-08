import argparse
import os
import pandas as pd
import json


def check_multipleSSM(patient_data):

    # Compute the VAF of each mutation
    patient_data['VAF'] = patient_data['t_alt_count']/patient_data['t_tot_count']

    # Keep the mutation with the max(mean VAF over regions))
    genes_list = patient_data.Hugo_Symbol.unique()
    for gene in genes_list:
        maxVAF = patient_data.loc[patient_data.Hugo_Symbol == gene, 'VAF'].max()
        patient_data = patient_data.drop(patient_data.loc[
            (patient_data.Hugo_Symbol == gene) & (patient_data.VAF < maxVAF)
            ].index)
    patient_data = patient_data.drop_duplicates(subset=['Hugo_Symbol'])
    patient_data.drop('VAF', axis=1, inplace=True)
    patient_data.reset_index(inplace=True, drop=True)
    return(patient_data)


def main():

    parser = argparse.ArgumentParser(
      description='Create input for PairTree from TCGA data',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--IntOgenfile', dest='IntOgenfile',
                        help='IntOgen file path')
    parser.add_argument('--input', dest='TCGA_folder',
                        help='File path to TCGA folder')
    parser.add_argument('txt_dic',
                        help='Directory to store the SNV txt_files')
    parser.add_argument('json_dic',
                        help='Directory to store the json files')
    parser.add_argument('--mode', dest='mode', default=1, type=int,
                        choices=[1, 2],
                        help=('mode=1:Keep only IntOgen genes \n'
                              + 'mode=2: Keep all the genes'))
    args = parser.parse_args()

    # Input
    IntOgenfile = args.IntOgenfile
    in_dirname = args.TCGA_folder
    txt_dirname = args.txt_dic
    json_dirname = args.json_dic
    mode = args.mode

    # Input
    # IntOgenfile = './LUAD/IntOGen-DriverGenes_LUAD_TCGA.tsv'
    # in_dirname = 'LUAD'
    # txt_dirname = 'txt'
    # json_dirname = 'json'
    # mode = 1

    # Create the outdir
    if not os.path.exists(os.path.join(os.getcwd(), txt_dirname)):
        os.makedirs(os.path.join(os.getcwd(), txt_dirname), exist_ok=True)
    if not os.path.exists(os.path.join(os.getcwd(), json_dirname)):
        os.makedirs(os.path.join(os.getcwd(), json_dirname), exist_ok=True)

    # IntOgendata
    IntOgendata = pd.read_csv(IntOgenfile, header=0, sep="\t")
    IntOgendata = pd.DataFrame({'Symbol': IntOgendata['Symbol']})

    # TCGA patients
    files = [
        filename for filename in os.listdir(in_dirname)
        if filename.startswith("TCGA")]

    for file in files:
        name = file.split('.')[0]
        try:
            patient_data = pd.read_csv(
                os.path.join(in_dirname, file), comment="#", sep="\t")
        except UnicodeDecodeError:
            continue

        if mode == 1:
            # 1.Keep only the IntOgen genes
            patient_data = patient_data.loc[
                patient_data['Hugo_Symbol'].isin(IntOgendata["Symbol"])]
        if patient_data.empty:
            print(name + "is empty")
            continue

        # 2.Keep only the non silent mutations
        patient_data = patient_data.loc[
            patient_data['Variant_Classification'] != "Silent"]
        if patient_data.empty:
            print(name + "is empty")
            continue

        # Filter columns
        patient_data['t_tot_count'] = (patient_data['t_alt_count']
                                       + patient_data['t_ref_count'])
        to_keep = ['Hugo_Symbol', 't_alt_count', 't_tot_count']
        patient_data = patient_data[to_keep]

        # For each gene keep the mutation with the largest VAF
        patient_data = check_multipleSSM(patient_data)

        # Create final txt file
        patient_data.rename(columns={'Hugo_Symbol': 'name',
                                     't_tot_count': 'total_reads',
                                     't_alt_count': 'var_reads'}, inplace=True)
        patient_data['var_read_prob'] = [0.5] * patient_data.shape[0]
        idx_list = patient_data.index.to_list()
        patient_data.insert(
            0, "id", ["{}{}".format("s", str(index)) for index in idx_list])
        outpath = os.path.join(os.getcwd(), txt_dirname)
        patient_data.to_csv("{}/{}{}".format(outpath, name, ".txt"),
                            sep='\t', index=False)

        # Write the json file
        details = {"samples": ['Sample 1'], "clusters": [], "garbage": []}
        details["clusters"] += [[s_i] for s_i in patient_data["id"]]
        outpath = os.path.join(os.getcwd(), json_dirname)
        with open("{}/{}{}".format(outpath, name, ".json"), "w") as json_f:
            json.dump(details, json_f)


if __name__ == '__main__':
    main()
