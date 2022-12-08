import os
import json
import pandas as pd


##############################################
#          Input Parameters                  #
##############################################

# If mode = 1: You keep only the IntOgen genes in the dataset.
# Then you run PairTree and create C-ToMeXo input
# If mode = 2: You keep all the genes in the dataset.
# Then you run PairTree. You prune the resulting trees such that
# only IntOgen genes are kept. Finally you create C-ToMeXo input
# number_of_trees_per_sample: how many tree structures you take for each sample
# with decreasing posterior probability

Pairtree_installation_path = '/proj/sc_ml/users/x_smadi/pairtree'

# If Preprocessing TCGA data : input_data -> the path to 
# the folder containing the TCGA samples
input_data = "./Biological_Data/LUAD"
cancer = 'LUAD'  # cancer type
IntOgen_file = './Biological_Data/IntOgen/IntOGen-DriverGenes_LUAD_TCGA.tsv'
output_folder = cancer + '/C_ToMeXo_input'
cluster_SNVS = False
mode = str(1)
number_of_trees_per_sample = str(1)

##############################################
#          Create Input for PairTree         #
##############################################
# Folders to save PairTree Input
txt_input = cancer + "/Pairtree_Input/txt"
json_input = cancer + "/Pairtree_Input/json"

# For TCGA
os.system("python ./lib/create_PairTree_input_TCGA.py --input "
          + input_data + " --IntOgenfile " + IntOgen_file + " " + txt_input
          + " " + json_input + " --mode " + mode)

# ##############################################
# #               Run PairTree                 #
# ##############################################

# Output Folders to store PairTee results
txt_output_dirname = './' + cancer + '/Pairtree_Output/txt'
json_output_dirname = './' + cancer + '/Pairtree_Output/json'
npz_dirname = './' + cancer + '/Pairtree_Output/npz'
html_dirname = './' + cancer + '/Pairtree_Output/html'

if not os.path.exists(os.path.join(os.getcwd(), txt_output_dirname)):
    os.makedirs(os.path.join(os.getcwd(), txt_output_dirname), exist_ok=True)

if not os.path.exists(os.path.join(os.getcwd(), json_output_dirname)):
    os.makedirs(os.path.join(os.getcwd(), json_output_dirname), exist_ok=True)

if not os.path.exists(os.path.join(os.getcwd(), html_dirname)):
    os.makedirs(os.path.join(os.getcwd(), html_dirname), exist_ok=True)

if not os.path.exists(os.path.join(os.getcwd(), npz_dirname)):
    os.makedirs(os.path.join(os.getcwd(), npz_dirname), exist_ok=True)

onlyfiles = [f for f in os.listdir(txt_input)]
for f in onlyfiles:
    name = f.split(".")[0]
    txt_f = os.path.join(txt_input, f)
    txt_fnew = os.path.join(txt_output_dirname, f)
    json_f = os.path.join(json_input, name + ".json")
    json_fnew = os.path.join(json_output_dirname, name + ".json")

    # Remove incorrect ploidy
    os.system("python " + Pairtree_installation_path
              + "/util/fix_bad_var_read_prob.py " + txt_f + " " + json_f + " "
              + txt_fnew + " " + json_fnew + " --action add_to_garbage")
    with open(json_fnew) as f:
        patient_data = json.load(f)
    if len(patient_data["garbage"]) == len(patient_data["clusters"]):
        print(json_fnew + "has all the SNVs in the garbage")
        continue
    # To find garbage relations you need many regions
    # if not len(patient_data['samples']) == 1:
    #     if not len(patient_data['clusters']) == 1:
    #         os.system("python " + Pairtree_installation_path
    #                   + "/bin/removegarbage " + txt_fnew + " " + json_fnew
    #                   + " " + json_fnew + ' --verbose')
    #         with open(json_fnew) as f:
    #             patient_data = json.load(f)
    #         if len(patient_data["clusters"]) == len(patient_data["garbage"]):
    #             print(json_fnew + " has all the SNVs in the garbage")
    #             continue

    npz_f = os.path.join(npz_dirname, name + ".npz")
    html_f = os.path.join(html_dirname, name + ".html")
    # Create the clusters
    if cluster_SNVS:
        os.system("python " + Pairtree_installation_path + "/bin/clustervars "
                  + txt_fnew + " " + json_fnew + " " + json_fnew
                  + " --chains 5 --model linfreq --concentration 0")
    else:
        # Bug of PairTree: It doesn't get rid of the garbage mutations
        # if you don't cluster them
        with open(json_fnew) as f:
            patient_data = json.load(f)
        if len(patient_data['garbage']) > 0:
            to_delete = []
            for C in patient_data['clusters']:
                if C[0] in patient_data['garbage']:
                    to_delete.append(C)
            patient_data['clusters'] = [
                x for x in patient_data['clusters'] if x not in to_delete]
            with open(json_fnew, "w") as f:
                json.dump(patient_data, f)

    # Run pairtree
    os.system("python " + Pairtree_installation_path + "/bin/pairtree "
              + txt_fnew + " " + npz_f + " --params " + json_fnew)
    # Create html
    os.system("python " + Pairtree_installation_path + "/bin/plottree "
              + txt_fnew + " " + json_fnew + " " + npz_f + " " + html_f)

##############################################
#      Create input for  C-ToMeXo            #
##############################################
os.system("python ./lib/reading_PairTree_Phylotrees.py "
          + txt_output_dirname + " " + json_output_dirname + " "
          + npz_dirname + " " + output_folder + " --IntOgenfile "
          + IntOgen_file + " --mode " + mode + " --NumberTrees "
          + number_of_trees_per_sample)

# Add the empty samples
samples= [
        filename.split('.')[0] for filename in os.listdir(input_data)
        if filename.startswith("TCGA")]
# Read IntOgendata
IntOgendata = pd.read_csv(IntOgen_file, header=0, sep="\t")
IntOgendata = pd.DataFrame({'Symbol': IntOgendata['Symbol']})

onlyfiles = [f[:-6] for f in os.listdir(os.path.join(output_folder, 'clones'))]
empty_samples = [f for f in samples if f not in onlyfiles]
for sample in empty_samples:
    # Create clone file
    pd.DataFrame([[0]*len(IntOgendata)], columns=IntOgendata.Symbol). to_csv(
        "{}/{}/{}".format(output_folder, 'clones', sample + "_0.csv"),
        index=False
    )
    # Create the weights
    pd.DataFrame([1]). to_csv(
        "{}/{}/{}".format(output_folder, 'weights', sample + "_0.csv"),
        index=False
    )
