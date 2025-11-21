from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

# definim graful in reteaua bayesiana
model = DiscreteBayesianNetwork([
    ('O', 'H'),
    ('O', 'W'),
    ('O', 'R'),
    ('H', 'R'),
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'E'),
    ('R', 'C')
])

# definim cpd

#P(o)
cpd_o = TabularCPD(variable='O', variable_card=2, 
                   values=[[0.3], [0.7]])
#P(h|O)
cpd_h = TabularCPD(variable='H', variable_card=2, 
                   values=[[0.1, 0.8],  
                           [0.9, 0.2]], 
                   evidence=['O'],
                   evidence_card=[2])

cpd_w = TabularCPD(variable='W', variable_card=2,
                   values=[[0.1, 0.4],   
                           [0.9, 0.6]],  
                   evidence=['O'],
                   evidence_card=[2])

cpd_r = TabularCPD(variable='R', variable_card=2,
                   values=[[0.4, 0.1, 0.1, 0.15, 0.1, 0.05, 0.05, 0.1], 
                           [0.6, 0.9, 0.9, 0.85, 0.9, 0.95, 0.95, 0.9]],
                   evidence=['H', 'W', 'O'],
                   evidence_card=[2, 2, 2])

cpd_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.2, 0.8, 0.2, 0.8],  
                           [0.8, 0.2, 0.8, 0.2]], 
                   evidence=['H', 'R'],
                   evidence_card=[2, 2])

cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.15, 0.60],  
                           [0.85, 0.40]], 
                   evidence=['R'],
                   evidence_card=[2])

model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)
print(model.check_model())