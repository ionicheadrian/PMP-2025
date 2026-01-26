import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy import stats

np.random.seed(74)


df = pd.read_csv('./bike_daily.csv')

# Prepare data
rentals = df['rentals'].values
temp_c = df['temp_c'].values
humidity = df['humidity'].values
wind_kph = df['wind_kph'].values
is_holiday = df['is_holiday'].values
season_cat = df['season'].astype('category').cat.codes.values

#a

# standarizarea variabilelor continue
temp_c_mean, temp_c_std = temp_c.mean(), temp_c.std()
humidity_mean, humidity_std = humidity.mean(), humidity.std()
wind_kph_mean, wind_kph_std = wind_kph.mean(), wind_kph.std()

temp_c = (temp_c - temp_c_mean) / temp_c_std
humidity = (humidity - humidity_mean) / humidity_std
wind_kph_std_vals = (wind_kph - wind_kph_mean) / wind_kph_std

print(f"\nParametrii de standardizare:")
print(f"Temperatura: mean={temp_c_mean:.2f}, std={temp_c_std:.2f}°C")
print(f"Umiditate: mean={humidity_mean:.4f}, std={humidity_std:.4f}")
print(f"Wind Speed: mean={wind_kph_mean:.2f} km/h, std={wind_kph_std:.2f} km/h")


# print(f"Temp_c_std: mean={temp_c.mean():.6f}, std={temp_c.std():.6f}")
# print(f"Humidity_std: mean={humidity.mean():.6f}, std={humidity.std():.6f}")
# print(f"Wind_kph_std: mean={wind_kph_std_vals.mean():.6f}, std={wind_kph_std_vals.std():.6f}")

# b

#construim modelul liniar

with pm.Model() as linear_model:
    intercept = pm.Normal('intercept', mu=0, sigma=100)
    beta_temp = pm.Normal('beta_temp', mu=0, sigma=50)
    beta_humidity = pm.Normal('beta_humidity', mu=0, sigma=50)
    beta_wind = pm.Normal('beta_wind', mu=0, sigma=50)
    

    m = intercept + beta_temp * temp_c + beta_humidity * humidity + beta_wind * wind_kph_std_vals
    #prior
    sigma = pm.HalfNormal('sigma', sigma=100)
    #likelihood
    rentals_obs = pm.Normal('rentals_obs', mu=m, sigma=sigma, observed=rentals)
    
    #facem sampling (posterior)
    trace_linear = pm.sample(2000, tune=1000, chains=4, random_seed=42, 
                             return_inferencedata=True, target_accept=0.95)
#rezumatul dupa posterior
print("\nrezumatul dupa posterior")
print(az.summary(trace_linear, var_names=['intercept', 'beta_temp', 'beta_humidity', 'beta_wind', 'sigma']))

# Plot posterior distributions
az.plot_trace(trace_linear, var_names=['intercept', 'beta_temp', 'beta_humidity', 'beta_wind', 'sigma'])
plt.tight_layout()
plt.savefig('./ex2b.png', dpi=300, bbox_inches='tight')

# c

#construim modelul 

temp_c2 = temp_c ** 2

with pm.Model() as poly_model:
    # priors pentru coeficienti
    intercept = pm.Normal('intercept', mu=0, sigma=100)
    beta_temp = pm.Normal('beta_temp', mu=0, sigma=50)
    beta_temp_sq = pm.Normal('beta_temp_sq', mu=0, sigma=50)
    beta_humidity = pm.Normal('beta_humidity', mu=0, sigma=50)
    beta_wind = pm.Normal('beta_wind', mu=0, sigma=50)
    
    
    m = (intercept + 
          beta_temp * temp_c + 
          beta_temp_sq * temp_c2 +
          beta_humidity * humidity + 
          beta_wind * wind_kph_std_vals)
    
    #prior pentru anomalii si noise
    sigma = pm.HalfNormal('sigma', sigma=100)
    
    #likelihood
    rentals_obs = pm.Normal('rentals_obs', mu=m, sigma=sigma, observed=rentals)
    
    trace_poly = pm.sample(2000, tune=1000, chains=4, random_seed=42, 
                           return_inferencedata=True, target_accept=0.95)
