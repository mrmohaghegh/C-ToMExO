import json
import os
import pandas as pd
import argparse
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from PhyloTree import *

def max_llh_tree(log_lik, numberTrees=1):
    # If n=1: Find the tree that has the max likelihood and return its position
    # If n!=1: Find the n trees with the highest log likelihood   

    if numberTrees==1:
        return([log_lik.argmax()])
    else:
        temp = pd.Series(log_lik)
        max_temp = temp.nlargest(numberTrees)
        llh_keys_max=max_temp.index.values.tolist()
        return(llh_keys_max)

def main():

    parser = argparse.ArgumentParser(
      description='Read PairTree PhyloTrees and create input for C-ToMeXo',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--IntOgenfile', dest='IntOgenfile',
      help='IntOgen file path')
    parser.add_argument('--NumberTrees', dest='NuTrees', default=1,
      help='The number of phylogenetic trees to be chosen for each sample')
    parser.add_argument('txt_dic',
      help='Directory of the txt files from PairTree')
    parser.add_argument('json_dic',
      help='Directory of the json files from PairTree')
    parser.add_argument('npz_dic',
      help='Directory of the npz files from PairTree')
    parser.add_argument('out_dic',
      help='Directory for storing the input of C-ToMeXo')
    args = parser.parse_args()
    

    #Parameters
    txt_dic = args.txt_dic
    json_dic=args.json_dic
    npz_dic = args.npz_dic
    intogenfile=args.IntOgenfile
    out_dic=args.out_dic
    out_clones= out_dic + "/clones"
    out_weights= out_dic + "/weights"
    numberTrees=args.NuTrees

    # numberTrees=1
    # cancer="LUSC"
    # intogenfile="IntOGen-DriverGenes_" + cancer +"_TCGA.tsv"
    # txt_dic = cancer + "_txt"
    # json_dic=cancer + "_json"
    # npz_dic = cancer + "_npz"
    # out_dic= cancer 
    # out_clones= cancer + "_clones"
    # out_weights= cancer + "_weights"
  
    if not os.path.exists(os.path.join(os.getcwd(),out_clones)):
        os.makedirs(os.path.join(os.getcwd(), out_clones), exist_ok=True)
    if not os.path.exists(os.path.join(os.getcwd(),out_weights)):
        os.makedirs(os.path.join(os.getcwd(),out_weights),exist_ok=True)
   
    #Read IntoGen genes
    IntOgendata = pd.read_csv(os.path.join(os.getcwd(),intogenfile), header = 0, sep="\t")['Symbol']
    IntOgendata = IntOgendata.to_list()
    #Output Initialisation
    file_list= [f for f in os.listdir(npz_dic) if os.path.isfile(os.path.join(npz_dic, f))]
    for file in file_list:
        name=file.split(".")[0]
        npz_f=os.path.join(npz_dic,file)
        txt_f = os.path.join(txt_dic, name +".txt")
        json_f=os.path.join(json_dic, name + ".json")

        #Load npz data
        data=np.load(npz_f)
        NodeNames=data['clustrel_posterior_vids.json'].decode()
        NodeNames=NodeNames.strip('[]\n').replace('"','').split(',')
        NodeNames=[x.lstrip() for x in NodeNames]
        structures=data['struct']
        posterior_prob=data['prob']
        cp=data['phi']

        #Create the Node:[ssm1,ssm2, ...] dictionary
        with open(json_f) as jsonFile:
            details = json.load(jsonFile)
        SampleNames=details['samples']
        node_to_ssm={}
        for idx, node in enumerate(NodeNames):
            node_to_ssm[node]=details['clusters'][idx]
        
        #Create the ssm:Gene dictionary
        txt_data=pd.read_csv(txt_f,sep="\t")
        ssm_to_gene={}
        for idx in range(txt_data.shape[0]):
            ssm_to_gene[txt_data.loc[idx, 'id']]= txt_data.loc[idx,'name']
        
        #Create the genes dictionary Node:[gene1, gene2 ,...]
        node_to_gene={"S0":[]}
        for key in node_to_ssm.keys():
            if (key!="S0"):
                node_to_gene[key]=[ssm_to_gene[ssm] for ssm in node_to_ssm[key]]
    
        #Find the best trees
        treekey=max_llh_tree(posterior_prob, numberTrees=numberTrees)
        
        #List of numberTrees best trees
        for idx, key in enumerate(treekey):
            sample_structure=structures[key]
            sample_structure=pd.DataFrame(sample_structure, index=NodeNames, columns=["parent"])
            sample_posterior=posterior_prob[key]
            sample_cp=pd.DataFrame(cp[key], index= ["S0"] + NodeNames,columns=SampleNames)
            
            #Average over the regions
            sample_cp=pd.DataFrame(sample_cp.mean(axis=1))

            tree=PhyloTree.from_pairtree(
                            structure=sample_structure,
                            node_to_gene=node_to_gene,
                            cp=sample_cp,
                            llh=sample_posterior)
            #tree=tree.prune_by_IntOgen(IntOgendata)
            #to_dot(tree, name, out_dic)

            if len(tree.weights())==0: # the tree include only the root
                print(name + "has an empty tree")
                continue
            else:
                tree.matrix(IntOgendata).to_csv("{}/{}".format(out_clones, name + "_" + str(idx) + ".csv"),index=False)
                tree.weights().to_csv("{}/{}".format(out_weights, name + "_" + str(idx) +".csv"),index=False)
                         
if __name__ == '__main__':
    main()

