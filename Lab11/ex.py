#ex1 pmp 
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

np.random.seed(74)

data=pd.read_csv("./Prices.csv")

#acum trebuie sa extragem datele necesare
#folosind .values transforma toata coloana din DataFrame intrun vector type numpy
pret=data["Price"].values          #ce vrem sa prezicem
viteza=data["Speed"].values         # prima varibila dependeta
log_hd=np.log(data["HardDrive"].values)     #a doua varibila dependenta, log natural (in enunt scrie hard disk :D )

with pm.Model() as model_weakly_informative:
    
    a=pm.Normal('alpha', mu=0,sigma=1000)
    #am inceput cu prior alfa
    #  slab informativ (sigma foarte mare)

    b1=pm.Normal("beta1", mu=0,sigma=1000)  #coeficientul speed
    b2=pm.Normal("beta2",mu=0,sigma=1000)   #coeficientulm ln(hd)

    sigma=pm.HalfNormal('sigma',sigma=1000)

    media_distributie= a + b1*viteza + b2*log_hd

    v=pm.Normal('verosimilitate', mu=media_distributie, sigma=sigma, observed=pret)
    trace_a = pm.sample(
        draws=1000, 
        tune=1000, 
        chains=2, 
        return_inferencedata=True, 
        random_seed=74
    )
    
print(az.summary(trace_a, var_names=['alpha', 'beta1', 'beta2', 'sigma']))