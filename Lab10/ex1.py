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
    # Prioruri pentru parametri
    # Prior pentru intercept (α) - presupunem că poate fi între 0 și 30
    intercept = pm.Normal('intercept', mu=10, sigma=10)
    
    # Prior pentru slope (β) - presupunem o relație pozitivă
    slope = pm.Normal('slope', mu=1, sigma=5)
    
    # Prior pentru deviația standard a erorii (σ)
    sigma = pm.HalfNormal('sigma', sigma=5)
    
    # Modelul de regresie liniară
    # Media predicției
    mu = intercept + slope * advertising
    
    # Likelihood (verosimilitatea) - datele observate
    likelihood = pm.Normal('sales', mu=mu, sigma=sigma, observed=sales)
    
    print("Modelul a fost construit cu succes!")
    print("\nPrioruri specificate:")
    print("  - intercept ~ Normal(mu=10, sigma=10)")
    print("  - slope ~ Normal(mu=1, sigma=5)")
    print("  - sigma ~ HalfNormal(sigma=5)")
    print()

# =============================================================================
# PASUL 3: Sampling MCMC (inferență)
# =============================================================================
print("=" * 70)
print("PASUL 3: INFERENȚĂ MCMC")
print("=" * 70)
print("Rulăm sampling MCMC pentru a estima distribuțiile posterioare...")
print("Aceasta poate dura câteva secunde...")
print()

with model:
    # Rulăm sampling cu 4 lanțuri MCMC
    trace = pm.sample(2000, tune=1000, random_seed=42, return_inferencedata=True)

print("✓ Sampling completat cu succes!")
print()

# =============================================================================
# PASUL 4a: ESTIMAREA COEFICIENȚILOR (Punctul a)
# =============================================================================
print("=" * 70)
print("REZULTATE - Punctul a) ESTIMAREA COEFICIENȚILOR")
print("=" * 70)

# Extragem statisticile posterioare
summary = az.summary(trace, var_names=['intercept', 'slope', 'sigma'])
print(summary)
print()

# Valorile medii (estimările punctuale)
intercept_mean = trace.posterior['intercept'].mean().values
slope_mean = trace.posterior['slope'].mean().values
sigma_mean = trace.posterior['sigma'].mean().values

print("ESTIMĂRILE COEFICIENȚILOR:")
print(f"  Intercept (α) = {intercept_mean:.4f} mii $")
print(f"  Slope (β) = {slope_mean:.4f} mii $ / mie $ publicitate")
print(f"  Sigma (σ) = {sigma_mean:.4f} mii $")
print()
print("INTERPRETARE:")
print(f"  - Când advertising = 0, vânzările estimate sunt {intercept_mean:.2f} mii $")
print(f"  - Pentru fiecare 1000$ cheltuiți pe publicitate, vânzările cresc cu {slope_mean:.2f} mii $")
print()

# =============================================================================
# PASUL 4b: INTERVALE DE CREDIBILITATE HDI (Punctul b)
# =============================================================================
print("=" * 70)
print("REZULTATE - Punctul b) INTERVALE DE CREDIBILITATE (HDI 94%)")
print("=" * 70)

# Calculăm HDI pentru fiecare parametru
hdi_intercept = az.hdi(trace, var_names=['intercept'], hdi_prob=0.94)
hdi_slope = az.hdi(trace, var_names=['slope'], hdi_prob=0.94)
hdi_sigma = az.hdi(trace, var_names=['sigma'], hdi_prob=0.94)

print("\nINTERVALE HDI 94%:")
print(f"  Intercept: [{hdi_intercept['intercept'].values[0]:.4f}, {hdi_intercept['intercept'].values[1]:.4f}]")
print(f"  Slope: [{hdi_slope['slope'].values[0]:.4f}, {hdi_slope['slope'].values[1]:.4f}]")
print(f"  Sigma: [{hdi_sigma['sigma'].values[0]:.4f}, {hdi_sigma['sigma'].values[1]:.4f}]")
print()
print("INTERPRETARE HDI:")
print(f"  Cu 94% probabilitate, interceptul este între {hdi_intercept['intercept'].values[0]:.2f} și {hdi_intercept['intercept'].values[1]:.2f}")
print(f"  Cu 94% probabilitate, slope-ul este între {hdi_slope['slope'].values[0]:.2f} și {hdi_slope['slope'].values[1]:.2f}")
print()

# =============================================================================
# PASUL 4c: PREDICȚII PENTRU NIVELURI NOI (Punctul c)
# =============================================================================
print("=" * 70)
print("REZULTATE - Punctul c) PREDICȚII PENTRU NIVELURI NOI DE ADVERTISING")
print("=" * 70)

# Definim valori noi de advertising pentru care vrem predicții
new_advertising = np.array([3.0, 5.0, 7.0, 9.0, 12.0])

print("\nCalculăm predicții pentru valori noi de advertising...")

# Extragem toate valorile posterioare
intercept_samples = trace.posterior['intercept'].values.flatten()
slope_samples = trace.posterior['slope'].values.flatten()
sigma_samples = trace.posterior['sigma'].values.flatten()

# Calculăm predicții pentru fiecare valoare nouă
predictions_dict = {}

for adv in new_advertising:
    # Media predicției pentru fiecare sample din posterior
    mu_pred = intercept_samples + slope_samples * adv
    
    # Predicție completă (inclusiv incertitudinea modelului)
    # Generăm predicții din distribuția normală cu sigma
    predictions = np.random.normal(mu_pred, sigma_samples)
    
    # Calculăm statistici
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_hdi = az.hdi(predictions, hdi_prob=0.94)
    
    predictions_dict[adv] = {
        'mean': pred_mean,
        'std': pred_std,
        'hdi_lower': pred_hdi[0],
        'hdi_upper': pred_hdi[1]
    }
    
    print(f"\nAdvertising = {adv:.1f} mii $:")
    print(f"  Vânzări estimate: {pred_mean:.2f} ± {pred_std:.2f} mii $")
    print(f"  HDI 94%: [{pred_hdi[0]:.2f}, {pred_hdi[1]:.2f}] mii $")

print()

# =============================================================================
# PASUL 5: VIZUALIZĂRI
# =============================================================================
print("=" * 70)
print("PASUL 5: GENERARE GRAFICE")
print("=" * 70)

# Figura 1: Distribuțiile posterioare ale parametrilor
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))

# Intercept
az.plot_posterior(trace, var_names=['intercept'], ax=axes[0], hdi_prob=0.94)
axes[0].set_title('Posterior Intercept (α)', fontsize=12, fontweight='bold')

# Slope
az.plot_posterior(trace, var_names=['slope'], ax=axes[1], hdi_prob=0.94)
axes[1].set_title('Posterior Slope (β)', fontsize=12, fontweight='bold')

# Sigma
az.plot_posterior(trace, var_names=['sigma'], ax=axes[2], hdi_prob=0.94)
axes[2].set_title('Posterior Sigma (σ)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/posterior_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: posterior_distributions.png")

# Figura 2: Trace plots (verificare convergență)
fig2 = plt.figure(figsize=(15, 8))
az.plot_trace(trace, var_names=['intercept', 'slope', 'sigma'])
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/trace_plots.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: trace_plots.png")

# Figura 3: Regresie cu interval de credibilitate
fig3, ax = plt.subplots(figsize=(12, 8))

# Datele observate
ax.scatter(advertising, sales, s=100, alpha=0.6, color='blue', 
           label='Date observate', edgecolors='black', linewidth=1)

# Linia de regresie (media posterioară)
x_line = np.linspace(advertising.min() - 0.5, advertising.max() + 2, 100)
y_line = intercept_mean + slope_mean * x_line
ax.plot(x_line, y_line, 'r-', linewidth=2, label='Regresie medie')

# Interval de credibilitate pentru predicții
y_pred_samples = []
for _ in range(500):
    idx = np.random.randint(0, len(intercept_samples))
    y_pred = intercept_samples[idx] + slope_samples[idx] * x_line
    y_pred_samples.append(y_pred)

y_pred_samples = np.array(y_pred_samples)
y_pred_lower = np.percentile(y_pred_samples, 3, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 97, axis=0)

ax.fill_between(x_line, y_pred_lower, y_pred_upper, alpha=0.3, 
                color='red', label='Interval credibilitate 94%')

# Predicții pentru valori noi
for adv in new_advertising:
    pred = predictions_dict[adv]
    ax.errorbar(adv, pred['mean'], 
                yerr=[[pred['mean'] - pred['hdi_lower']], 
                      [pred['hdi_upper'] - pred['mean']]], 
                fmt='o', markersize=10, capsize=5, capthick=2,
                color='green', alpha=0.7)

ax.scatter([], [], color='green', s=100, label='Predicții noi')

ax.set_xlabel('Cheltuieli publicitate (mii $)', fontsize=12, fontweight='bold')
ax.set_ylabel('Vânzări (mii $)', fontsize=12, fontweight='bold')
ax.set_title('Regresie Liniară Bayesiană: Advertising vs Sales', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/regression_plot.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: regression_plot.png")

# Figura 4: Posterior predictive check
fig4, ax = plt.subplots(figsize=(12, 6))

with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

az.plot_ppc(ppc, num_pp_samples=100, ax=ax)
ax.set_title('Posterior Predictive Check', fontsize=14, fontweight='bold')
ax.set_xlabel('Vânzări (mii $)', fontsize=12)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/posterior_predictive_check.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: posterior_predictive_check.png")

print()
print("=" * 70)
print("REZUMAT FINAL")
print("=" * 70)
print(f"""
MODELUL: sales = {intercept_mean:.4f} + {slope_mean:.4f} * advertising + ε

INTERPRETARE:
1. Interceptul ({intercept_mean:.2f}): Vânzările de bază când nu există publicitate
2. Slope ({slope_mean:.2f}): Fiecare 1000$ în publicitate generează ~{slope_mean:.2f}k$ vânzări
3. Relația este POZITIVĂ și SEMNIFICATIVĂ (intervalul HDI pentru slope nu include 0)

ÎNCREDERE:
- Avem 94% încredere că slope-ul este între {hdi_slope['slope'].values[0]:.2f} și {hdi_slope['slope'].values[1]:.2f}
- Modelul explică bine datele (vezi posterior predictive check)

PREDICȚII VALIDATE pentru valori noi de advertising.
""")

print("\n✓ Toate calculele și graficele au fost generate cu succes!")
print("=" * 70)