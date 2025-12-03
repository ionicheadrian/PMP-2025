import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
advertising = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 
                        6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])

sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])
print(f"Nr de observatii {len(advertising)}")
print(f"Rangeul de sales {sales.min():1f} - {sales.min():1f}")
print(f"range ads: {advertising.min():1f} - {advertising.min():1f}")


with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=10, sigma=10)
    slope = pm.Normal('slope', mu=1, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)
    mu = intercept + slope * advertising
    likelihood = pm.Normal('sales', mu=mu, sigma=sigma, observed=sales)


with model:
    trace = pm.sample(2000, tune=1000, random_seed=42, return_inferencedata=True)

print("sampling completat")
print()
summary = az.summary(trace, var_names=['intercept', 'slope', 'sigma'])
print(summary)
print()

intercept_mean = trace.posterior['intercept'].mean().values
slope_mean = trace.posterior['slope'].mean().values
sigma_mean = trace.posterior['sigma'].mean().values

print(f"intercept {intercept_mean:4f}")
print(f"slope {slope_mean:4f}")
print(f"sigma {sigma_mean:4f}")

print(f"când ads = 0,vanzarile estimate sunt {intercept_mean:.2f} mii $")
print(f"pt fiecare 1000$ dati pe ads, vanzarile cresc cu {slope_mean:.2f}")
print()


# b)


hdi_intercept = az.hdi(trace, var_names=['intercept'], hdi_prob=0.94)
hdi_slope = az.hdi(trace, var_names=['slope'], hdi_prob=0.94)
hdi_sigma = az.hdi(trace, var_names=['sigma'], hdi_prob=0.94)

print(f"Intercept: [{hdi_intercept['intercept'].values[0]:.4f}, {hdi_intercept['intercept'].values[1]:.4f}]")
print(f"Slope: [{hdi_slope['slope'].values[0]:.4f}, {hdi_slope['slope'].values[1]:.4f}]")
print(f"Sigma: [{hdi_sigma['sigma'].values[0]:.4f}, {hdi_sigma['sigma'].values[1]:.4f}]")
print()


new_advertising = np.array([3.0, 5.0, 7.0, 9.0, 12.0])

intercept_samples = trace.posterior['intercept'].values.flatten()
slope_samples = trace.posterior['slope'].values.flatten()
sigma_samples = trace.posterior['sigma'].values.flatten()

predictions_dict = {}

for adv in new_advertising:

    mu_pred = intercept_samples + slope_samples * adv
    predictions = np.random.normal(mu_pred, sigma_samples)
    

    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_hdi = az.hdi(predictions, hdi_prob=0.94)
    
    predictions_dict[adv] = {
        'mean': pred_mean,
        'std': pred_std,
        'hdi_lower': pred_hdi[0],
        'hdi_upper': pred_hdi[1]
    }
    
    print(f"\nAdvertising = {adv:.1f}:")
    print(f"vanzari estimate: {pred_mean:.2f} +/- {pred_std:.2f}")
    print(f"HDI 94%: [{pred_hdi[0]:.2f}, {pred_hdi[1]:.2f}]")
