import argparse
from math import ceil
import os
import pandas as pd
import json
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CNA_correction_cases import *

def calculate_wmega(patient_data, regions, ASCATInfo, male, thres, prior):
    
    wmega_info=pd.DataFrame([], columns=[reg + "wmega" for reg in regions],
                                    index=[gene for gene in patient_data.Hugo_Symbol])
    list_of_genes=patient_data.Hugo_Symbol.to_list()
    for gene in list_of_genes:
        #Initialisation
        log_lik=pd.DataFrame([], index=["Case1","Case2","Case3"], columns=regions)
        ml_wmega = pd.DataFrame([], index=["Case1","Case2","Case3"], columns=regions)
        #Gene info
        patient_data_gene=patient_data.loc[patient_data["Hugo_Symbol"]==gene].reset_index(drop=True)     
        chrom=patient_data_gene.chr[0]
        start=patient_data_gene.start[0]
        end=patient_data_gene.stop[0]
        for reg in regions:
            ASCATInfo_gene=ASCATInfo.loc[ASCATInfo["sample"]== patient_data_gene.SampleID[0] + "-" + reg]
            ASCATInfo_gene=ASCATInfo_gene.loc[ASCATInfo_gene["chr"]==chrom]
            ASCATInfo_gene=ASCATInfo_gene.loc[(ASCATInfo_gene['startpos']< start) & 
                                                (ASCATInfo_gene['endpos']> start) & 
                                                (ASCATInfo_gene['startpos']< end) & 
                                                (ASCATInfo_gene['endpos']> end)]
            if ASCATInfo_gene.shape[0]==0: #Not such bin
                log_lik.drop(reg, axis = 1, inplace=True)
                ml_wmega.loc['Case1', reg]=float("nan")
                ml_wmega.loc['Case2', reg]=float("nan")
                ml_wmega.loc['Case3', reg]=float("nan")
                continue 
            ASCATInfo_gene=ASCATInfo_gene.reset_index(drop=True)
            assert ASCATInfo_gene.shape[0] == 1, "Multiple CNA in gene " + gene + " and sample " + patient_data.SampleID.unique() + "-" + reg + ", ASCAT ERROR!!" 
            CN_major= ASCATInfo_gene.nMajor[0]
            CN_minor=ASCATInfo_gene.nMinor[0]
            if  (male==True) and (int(chrom)==23 or int(chrom)==24):
                CN_normal=1
            else:
                CN_normal=2
            ACF=ASCATInfo_gene.ACF[0]
            CN_Tot= CN_normal + (CN_major + CN_minor - CN_normal)*ACF
            Tcount=patient_data_gene[reg + "Tcount"][0]
            Vcount=patient_data_gene[reg + "Vcount"][0]
            
            # CASE 1
            ml_wmega.loc['Case1', reg], log_lik.loc['Case1', reg]=case1(Tcount, Vcount, CN_Tot, ACF)

            #CASE 2
            ml_wmega.loc['Case2', reg], log_lik.loc['Case2', reg]=case2(CN_minor, Tcount, Vcount, CN_Tot, ACF)

            #CASE 3
            ml_wmega.loc['Case3', reg], log_lik.loc['Case3', reg] = case2(CN_major, Tcount, Vcount, CN_Tot, ACF)

        
        if log_lik.shape[1]==0: #all regions have been dropped due to ASCAT
            wmega_info=wmega_info.drop(gene)
            patient_data=patient_data.drop(patient_data.loc[patient_data.Hugo_Symbol==gene].index).reset_index(drop=True)
            continue

        No_regions=log_lik.shape[1] #regions may have deleted due to ASCAT info
        log_lik=log_lik.astype(float).sum(axis=1)
        #Check with the threshold for garbage mutations 
        if log_lik.max() < thres*No_regions :
            wmega_info=wmega_info.drop(gene)
            patient_data=patient_data.drop(patient_data.loc[patient_data.Hugo_Symbol==gene].index).reset_index(drop=True)
        else: #choose best case
            log_lik = log_lik + log_prior(prior)
            chosen_case=log_lik.idxmax()
            for reg in regions:
                wmega_info.loc[gene, reg + "wmega"]=ml_wmega.loc[chosen_case ,reg]
            if wmega_info.loc[gene].isnull().values.any(): #Fixing NaN wmega
                _mean_wmega=wmega_info.loc[gene].dropna().mean()
                wmega_info.loc[gene].fillna(_mean_wmega, inplace=True)
            #Fixing wmega>1 or wmega==0
            for reg in regions:
                if wmega_info.loc[gene, reg + "wmega"] > 1:
                    wmega=wmega_info.loc[gene, reg + "wmega"]
                    Tcount=patient_data_gene[reg+"Tcount"][0]
                    #Changing:
                    wmega_info.loc[gene, reg + "wmega"]=1.0
                    patient_data.loc[patient_data.Hugo_Symbol==gene, reg+"Tcount"] = ceil(Tcount*wmega)
                if  wmega_info.loc[gene, reg + "wmega"] == 0:
                    wmega_info.loc[gene, reg + "wmega"]= 0.0001
    wmega_info.reset_index(drop=True, inplace=True)
    patient_data=pd.concat([patient_data,wmega_info], axis=1)
    return(patient_data)

def check_multipleSSM(patient_data, regions):
    
    #Compute the VAF of each mutation
    VAF_per_region=pd.DataFrame(
      [patient_data[reg + "Vcount"]/patient_data[reg + "Tcount"] for reg in regions]).transpose()
    patient_data['VAF']=VAF_per_region.mean(axis=1)
    
    #Keep the mutation with the max(mean VAF over regions))
    genes_list=patient_data.Hugo_Symbol.unique()
    for gene in genes_list:
      maxVAF=patient_data.loc[patient_data.Hugo_Symbol==gene]['VAF'].max()
      patient_data=patient_data.drop(patient_data.loc[(patient_data.Hugo_Symbol == gene) & (patient_data.VAF < maxVAF)].index)

    #Delete VAF column
    patient_data.drop('VAF', axis = 1, inplace=True)
    return(patient_data)


def main():

    parser = argparse.ArgumentParser(
      description='Create input for PairTree from TracerXdata',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--IntOgenfile', dest='IntOgenfile',
      help='IntOgen file path')
    parser.add_argument('--TracerXfile', dest='in_file',
      help='TracerX file path')
    parser.add_argument('txt_dic',
      help='Directory to store the SNV txt_files')
    parser.add_argument('json_dic',
      help='Directory to store the json files')
    parser.add_argument('--garbage_likel_thr', dest='thres', type=int, default=-50,
        help='Likelihood threshold for declaring a mutation garbage')  
    parser.add_argument('--prior_cases', dest='prior', type=float, default=0.001,
        help='Parameter for calculating the prior probability in each case') 
    args = parser.parse_args()
    
    #Parameters
    in_file =args.in_file
    out_dirname= args.txt_dic
    json_dirname=args.json_dic
    IntOgenfile=args.IntOgenfile
    thres=args.thres
    prior=args.prior
    
    #Cancer description
    if ('LUAD' in IntOgenfile):
        cancer_description="Invasive adenocarcinoma"
    else:
        cancer_description="Squamous cell carcinoma"

    #Create the outdir
    if not os.path.exists(os.path.join(os.getcwd(), out_dirname)):
        os.makedirs(os.path.join(os.getcwd(), out_dirname), exist_ok=True)
    if not os.path.exists(os.path.join(os.getcwd(), json_dirname)):
        os.makedirs(os.path.join(os.getcwd(), json_dirname), exist_ok=True)

    # IntOgendata
    IntOgendata = pd.read_csv(IntOgenfile, header = 0, sep="\t")
    IntOgendata=pd.DataFrame({'Symbol' : IntOgendata['Symbol']})

    # TRACERX Data
    TracerXdata=pd.read_excel(in_file,sheet_name="TableS3",comment="#",header=0)
    SampleInfo=pd.read_excel(in_file,sheet_name="TableS2",comment="#",header=0)
    SampleInfo=SampleInfo.loc[SampleInfo['Histology']==cancer_description]
    ASCATInfo=pd.read_excel(in_file,sheet_name="TableS10",comment="#",header=0)
    ASCATInfo = ASCATInfo[ASCATInfo.cnTotal != 0].reset_index(drop=True)
    
    #1.Confirm the sample IDs and keep only the ones for the specific cancer type
    TracerXdata=TracerXdata.loc[TracerXdata['SampleID'].isin(SampleInfo["TRACERxID"])]
    
    #2.Keep only the IntOgen genes
    TracerXdata=TracerXdata.loc[TracerXdata['Hugo_Symbol'].isin(IntOgendata["Symbol"])] 

    #3.Keep only the exonic mutations
    TracerXdata=TracerXdata.loc[(TracerXdata['func']=="exonic") | (TracerXdata['func']=="exonic;splicing") | 
                                                                    (TracerXdata['func']=="splicing")] 

    #4.Remove Synonymous mutations
    TracerXdata=TracerXdata.loc[TracerXdata['exonic.func']!="synonymous SNV"].reset_index(drop=True)

    #5.Remove rows that do not have genes but dates
    to_keep=[x for x in TracerXdata.index.to_list() if type(TracerXdata['Hugo_Symbol'][x])==str]
    TracerXdata = TracerXdata.iloc[to_keep]
    
    #Filter columns
    to_keep=["SampleID","chr","start","stop","Hugo_Symbol","RegionSum"] 
    TracerXdata=TracerXdata[to_keep]

    #Define tumor Samples
    sample_list=TracerXdata.SampleID.unique()

    #You don't have ASCAT info for all regions. 
    #The regions that you do not have ASCAT info will be excluded.
    Excluded_Regions=[]

    for sample in sample_list:
    
        patient_data=TracerXdata.loc[TracerXdata['SampleID']==sample].reset_index(drop=True) 
        
        #Find if the sample is male
        if SampleInfo.loc[SampleInfo['TRACERxID']==sample]['Gender'].str.contains('Female').any():
            male=False
        else:
            male=True
        
        #Find the regions that you have in the sample
        regions=[val.split(':')[0] for val in patient_data.loc[0,'RegionSum'].split(';')]
        
        #Expand the RegionSum column 
        patient_data[regions] =patient_data['RegionSum'].str.split(pat=";", expand=True)

        to_delete=[]
        for reg in regions:
            samp_reg= sample + "-" + reg
            #Delete regions from lymph nodes
            if "LN" in reg:
                Excluded_Regions.append(samp_reg)
                patient_data.drop(reg, axis = 1, inplace=True)
                to_delete.append(reg)
                continue
            #Delete regions that do not have ASCAT info    
            if samp_reg not in list(ASCATInfo["sample"]):
                Excluded_Regions.append(samp_reg)
                patient_data.drop(reg, axis = 1, inplace=True)
                to_delete.append(reg)
        regions= [reg for reg in regions if reg not in to_delete]

        #Remove Ri
        for reg in regions:
            patient_data[[reg + "name", reg +"Vcount", reg +"Tcount"]] = patient_data[reg].str.split(pat=":|/", expand=True)
        to_delete=["RegionSum"] + [reg + "name" for reg in regions] + [reg for reg in regions]
        patient_data.drop(to_delete, axis = 1, inplace=True)

        #Transfrm the columns to int
        to_int=lambda x:{x + "Vcount" : int, x + "Tcount" : int}
        columns_to_int={ }
        for reg in regions:
            columns_to_int.update(to_int(reg))
        patient_data=patient_data.astype(columns_to_int, copy=False)

        #For each gene keep the mutation with the greatest mean VAF
        patient_data=check_multipleSSM(patient_data,regions).reset_index(drop=True)

        #Calculate wmega
        patient_data=calculate_wmega(patient_data, regions, ASCATInfo, male, thres, prior)

        #Change everything to string again
        to_str=lambda x:{x + "Vcount" : str, x + "Tcount" : str, x + "wmega" : str }
        columns_to_str={ }
        for reg in regions:
            columns_to_str.update(to_str(reg))
        patient_data=patient_data.astype(columns_to_str,copy=False)
 
        #Create the final dataset
        patient_data.rename(columns={'Hugo_Symbol': 'name'}, inplace=True)
        patient_data["var_reads"]=patient_data[[reg + "Vcount" for reg in regions]].agg(','.join, axis=1)
        patient_data["total_reads"]=patient_data[[reg + "Tcount" for reg in regions]].agg(','.join, axis=1)
        patient_data["var_read_prob"]=patient_data[[reg + "wmega" for reg in regions]].agg(','.join, axis=1) 
        patient_data=patient_data[["name", "var_reads", "total_reads", "var_read_prob" ]]
        patient_data.reset_index(inplace=True,drop=True)
        patient_data.insert(0,"id",["{}{}".format("s",str(index)) for index in patient_data.index.to_list()])

        #Write the file
        outpath=os.path.join(os.getcwd(),out_dirname)
        patient_data.to_csv("{}/{}{}".format(outpath,sample,".txt"),sep='\t', index=False)
        
        #Write the json file
        details={"samples" : [], "clusters" : [], "garbage" : []}
        details["clusters"] += [[s_i] for s_i in patient_data["id"]]
        details["samples"] += ["{} {}".format("Sample",reg[1:]) for reg in regions]
        outpath=os.path.join(os.getcwd(), json_dirname)
        with open("{}/{}{}".format(outpath,sample,".json"),"w") as json_file:
            json.dump(details, json_file)
    
    #Save the excluded regions
    pd.DataFrame(Excluded_Regions, columns=[cancer_description]).to_csv(cancer_description +"-Excluded_Regions.txt", sep='\t', index=False)
    
if __name__ == '__main__':
    main()        



