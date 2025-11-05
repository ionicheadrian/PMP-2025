import numpy as np
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt
    
a_post = 181  # a_prior + suma_apeluri = 1 + 180
b_post = 10.1  # b_prior + numar_ore = 0.1 + 10

samples = stats.gamma.rvs(a=a_post, scale=1/b_post, size=10000)

az.plot_posterior(samples, hdi_prob=0.94, 
                  point_estimate='mode',
                  ref_val=18) 
plt.title('Distributia post pt Lambda')
plt.xlabel('apeluri / ora')
plt.show()

hdi = az.hdi(samples, hdi_prob=0.94)
mode = (a_post - 1) / b_post

print(f"HDI 94%: [{hdi[0]:.2f}, {hdi[1]:.2f}]")
print(f"Mode: {mode:.2f}")