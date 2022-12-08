import numpy as np
import pandas as pd
from scipy.special import betainc, beta, gammaln, betaln
from scipy.stats import binom
import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CNA_correction_cases import *


def main():

    parser = argparse.ArgumentParser(
      description='Check different options for CNA correction',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('TracerXdata',
        help='File path for TRACERx data')
    parser.add_argument('patient_id',
        help='The id of the patient in TRACERx')
    parser.add_argument('gene',
        help='Gene of interest')
    parser.add_argument('--case1 ',dest='case1', type=int,
        choices=[1,2],
        help='Option for case 1')
    parser.add_argument('--case2 ',dest='case2',type=int,
        choices=[1,2,3],
        help='Option for case 2')
    parser.add_argument('--case3 ',dest='case3',type=int,
        choices=[1,2,3],
        help='Option for case 3')
    parser.add_argument('--prior_cases', dest='prior', type=float, default=0.001,
        help='Threshold for declaring a mutation garbage') 
    args = parser.parse_args()

    #Main information-> Will become input in the script 
    data=args.TracerXdata
    patient_id =args.patient_id
    gene=args.gene
    case_1_option=args.case1
    case_2_option=args.case2
    case_3_option=args.case3
    prior=args.prior

    #Read the data
    TracerXdata=pd.read_excel(data,sheet_name="TableS3",comment="#",header=0)
    ASCATInfo=pd.read_excel(data,sheet_name="TableS10",comment="#",header=0)
    ASCATInfo = ASCATInfo[ASCATInfo.cnTotal != 0].reset_index(drop=True)    

    #Sample Info for male or not male
    SampleInfo=pd.read_excel(data, sheet_name="TableS2", comment="#",header=0)
    if SampleInfo.loc[SampleInfo['TRACERxID']==patient_id]['Gender'].str.contains('Female').any():
        male=False
    else:
        male=True

    #Filter columns
    to_keep=["SampleID","chr","start","stop","Hugo_Symbol","RegionSum"] 
    TracerXdata=TracerXdata[to_keep]

    #Keep only the patient and gene
    TracerXdata_gene=TracerXdata.loc[(TracerXdata['SampleID']==patient_id) & 
                                (TracerXdata["Hugo_Symbol"]==gene)].reset_index(drop=True) 
    if TracerXdata_gene.shape[0]== 0:
        print("Gene " + gene + " is not mutated in patient " + patient_id)

    #Find the regions
    regions=[val.split(':')[0] for val in TracerXdata_gene['RegionSum'][0].split(';')]

    #Expand the RegionSum column 
    TracerXdata_gene[regions] =TracerXdata_gene['RegionSum'].str.split(pat=";", expand=True)

    to_delete=[]
    for reg in regions:
        samp_reg= patient_id + "-" + reg
        #Delete regions that do not have ASCAT info    
        if samp_reg not in list(ASCATInfo["sample"]):
            TracerXdata_gene.drop(reg, axis = 1, inplace=True)
            to_delete.append(reg)
    regions= [reg for reg in regions if reg not in to_delete]
    if len(regions)==0:
        print("Gene " + gene + " has not Copy Number Information in any region in " + patient_id)

    #Remove Ri
    for reg in regions:
        TracerXdata_gene[[reg + "name", reg +"Vcount", reg +"Tcount"]] = TracerXdata_gene[reg].str.split(pat=":|/", expand=True)
    to_delete=["RegionSum"] + [reg + "name" for reg in regions] + [reg for reg in regions]
    TracerXdata_gene.drop(to_delete, axis = 1, inplace=True)

    #Change them to integers
    to_int=lambda x:{x+"Vcount" : int, x+"Tcount" : int}
    columns_to_int={ }
    for reg in regions:
        columns_to_int.update(to_int(reg))
    TracerXdata_gene=TracerXdata_gene.astype(columns_to_int,copy=False)

    #Start the printing
    to_print = "Number of mutations in the gene: " + str(TracerXdata_gene.shape[0]) +"\n"

    for mut in range(TracerXdata_gene.shape[0]):
        to_print += "Mutation " + str(mut) +":\n" 
        chrom=TracerXdata_gene.chr[mut]
        start=TracerXdata_gene.start[mut]
        end=TracerXdata_gene.stop[mut]
        results=pd.DataFrame([], index=["Case1","Case2","Case3"], columns=regions)
        for reg in regions:
            ASCATInfo_gene=ASCATInfo.loc[ASCATInfo["sample"]== patient_id + "-" + reg]
            ASCATInfo_gene=ASCATInfo_gene.loc[ASCATInfo_gene["chr"]==chrom]
            ASCATInfo_gene=ASCATInfo_gene.loc[(ASCATInfo_gene['startpos']< start) & 
                                                (ASCATInfo_gene['endpos']> start) & 
                                                (ASCATInfo_gene['startpos']< end) & 
                                                (ASCATInfo_gene['endpos']> end)]
            if ASCATInfo_gene.shape[0]==0: #Not such bin
                results[reg]=None
                continue 
            ASCATInfo_gene=ASCATInfo_gene.reset_index(drop=True)
            assert ASCATInfo_gene.shape[0] == 1, "Multiple CNA in gene " + gene + " and sample " + patient_id + "-" + reg + ", ASCAT ERROR!!" 
            CN_major= ASCATInfo_gene.nMajor[0]
            CN_minor=ASCATInfo_gene.nMinor[0]
            if  (male==True) and (int(chrom)==23 or int(chrom)==24):
                CN_normal=1
            else:
                CN_normal=2
            ACF=ASCATInfo_gene.ACF[0]
            CN_Tot= CN_normal + (CN_major + CN_minor - CN_normal)*ACF
            Tcount=TracerXdata_gene[reg+"Tcount"][mut]
            Vcount=TracerXdata_gene[reg+"Vcount"][mut]
            
            # CASE 1
            _ , results.loc['Case1', reg]=case1(Tcount, Vcount, CN_Tot, ACF, case_1_option)

            #CASE 2
            _ , results.loc['Case2', reg]=case2(CN_minor, Tcount, Vcount, CN_Tot, ACF, case_2_option)

            #CASE 3
            _ , results.loc['Case3', reg]=case2(CN_major, Tcount, Vcount, CN_Tot, ACF, case_3_option)

        results=results.astype(float).round(3)
        results['Posterior']=results.sum(axis=1)+ log_prior(prior)   
        to_print += "Log_likelihood of Case1 with option " + str(case_1_option) + " for all regions:\n"
        to_print += str(np.array(results.loc['Case1',results.columns!='Posterior'])) +"\n"
        to_print += "Log_likelihood of Case2 with option " + str(case_2_option) + " for all regions:\n"
        to_print += str(np.array(results.loc['Case2',results.columns!='Posterior'])) +"\n"
        to_print += "Log_likelihood of Case3 with option " + str(case_3_option) + " for all regions:\n"
        to_print += str(np.array(results.loc['Case3',results.columns!='Posterior'])) +"\n"
        to_print += "Posterior of the different cases: \n" 
        to_print += str(np.array(results.loc[:,'Posterior'])) +"\n"
        to_print += "\n"
        to_print += "where None means that there are not CNA info for the region\n"
        to_print += "\n"

    print(to_print)

if __name__ == '__main__':
    main() 


             
            






