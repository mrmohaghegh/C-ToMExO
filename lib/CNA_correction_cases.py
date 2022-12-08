import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, gammaln, betaln, beta 


def log_prior(p):
    return([np.log(1-2*p),np.log(p), np.log(p)])


def unnormalized_beta(x,a,b):
    if (beta(a,b)==0) | (betainc(a,b,x)==0):
        return(- np.inf)
    else:
        return(betaln(a, b) + np.log(betainc(a, b, x)))

def tchoosev(t,v):
    return(gammaln(t+1)-gammaln(v+1)-gammaln(t-v+1))

def case1(Tcount, Vcount, CN_Tot, ACF, option=1):
    Q=1
    wmega=Q/CN_Tot 
    if option==2: #best phi
        ml_phi=np.minimum(Vcount/(Tcount*wmega), ACF)
        if ml_phi!=0:
            return(round(wmega,4), tchoosev(Tcount,Vcount) + Vcount*np.log(ml_phi*wmega) + (Tcount-Vcount)*np.log(1-ml_phi*wmega))
        else: #Vcount = 0
            #return(np.log(np.exp(tchoosev(Tcount,Vcount)*0**Vcount*1**Tcount)))
            return(round(wmega,4), 0)
    elif option==1: #integrate over phi
        return(round(wmega,4), -np.log(ACF*wmega) + tchoosev(Tcount,Vcount) + unnormalized_beta(ACF*wmega,Vcount+1,Tcount-Vcount+1))   
    else:
        print("Invalid option for case1.\n")
        print("Option should take one of the values [1,2]")

def case2(multiplicity, Tcount, Vcount, CN_Tot, ACF, option=1):
    if option==3: #equal to purity
        if (multiplicity==0) & (Vcount!=0):
            return(0, - np.inf)
        elif (multiplicity==0) & (Vcount==0):
            return(0, 0)
        else:
            wmega=multiplicity/CN_Tot
            return(round(wmega,4), tchoosev(Tcount,Vcount)+ Vcount*np.log(ACF* wmega) + (Tcount-Vcount)*np.log(1-ACF*wmega))
    elif option==2: #best phi
        ml_phi=np.minimum(np.maximum(CN_Tot*Vcount/Tcount-(multiplicity-1)*ACF, ACF)
                ,1)
        ml_Q= 1+(multiplicity-1)*(ACF/ml_phi)
        mutation_prob=(ml_phi+(multiplicity-1)*ACF)/CN_Tot
        if (mutation_prob==0) & (Vcount!=0):
            return(round(ml_Q/CN_Tot,4), - np.inf)
        if (mutation_prob==0) & (Vcount==0):
            return(round(ml_Q/CN_Tot,4), 0)
        else:
            return(round(ml_Q/CN_Tot,4), tchoosev(Tcount, Vcount) + Vcount*np.log(mutation_prob) + (Tcount-Vcount)*np.log(1-mutation_prob))
    elif option==1: #integrate
        def to_integrate(x):
            return(x**Vcount)*(1-x)**(Tcount-Vcount)
        start=(multiplicity*ACF)/CN_Tot
        end=(1+(multiplicity-1)*ACF)/CN_Tot
        if ACF==1:
            log_lik= - np.inf
        else:
            integral= quad(to_integrate,start,end)[0]
            if integral==0: #numerical problem
                log_lik= -np.inf
            else:
                log_lik= np.log(CN_Tot/(1-ACF)) + tchoosev(Tcount, Vcount) + np.log(integral)
        ml_phi=np.minimum(np.maximum(CN_Tot*Vcount/Tcount-(multiplicity-1)*ACF, ACF)
                ,1)
        ml_Q= 1+(multiplicity-1)*(ACF/ml_phi)
        return(round(ml_Q/CN_Tot,4),log_lik)
    else:
        print("Invalid option for case2/case3\n")
        print("Option should take one of the values [1,2,3]")
