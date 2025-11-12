import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def run_bayesian_analysis():
    # date de observatie
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

    #definim modelul , x fiind priorul informativ din date
    x = np.mean(data)

    print(f"x = {x}")

    # Modelul Bayesian cu prior-uri slabe
    with pm.Model() as model_weak:

        #prior pentru mu si sigma
        #avem ca media este x cu siguranta sigma [x-sigma,x+sigma]
        m = pm.Normal('m', mu=x, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        #likelyhood 
        y = pm.Normal('y', mu=m, sigma=sigma, observed=data)
        
        # Sampling cores pt Windows 
        weak = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=74, cores=1)

    summary_weak = az.summary(weak, hdi_prob=0.95)
    print(summary_weak)


    # punctu b
    print("\nHDI 95%:")
    print(f"media reala de date este: [{summary_weak.loc['m', 'hdi_2.5%']:.2f}, {summary_weak.loc['m', 'hdi_97.5%']:.2f}]")
    print(f"derivatia reala este : [{summary_weak.loc['sigma', 'hdi_2.5%']:.2f}, {summary_weak.loc['sigma', 'hdi_97.5%']:.2f}]")

    

    mean_freq = np.mean(data)
    std_freq = np.std(data, ddof=1)  # ddof=1 pentru sample std

    # luam estimarile bayesiene, media posteiorului 
    mean_bayes = weak.posterior['m'].mean().values
    std_bayes = weak.posterior['sigma'].mean().values

    print(f"mediat pt")
    print(f"frequentist: {mean_freq:.2f}")
    print(f"bayesian:    {mean_bayes:.2f}")
    print("\n")
    print(f"\ndeviatia standard")
    print(f"frequentist: {std_freq:.2f}")
    print(f"bayesian:    {std_bayes:.2f}")
    print("\n")
    print("\ndiferentele")
    print(f"mu = {abs(mean_bayes - mean_freq):.4f}")
    print(f"sigma= {abs(std_bayes - std_freq):.4f}")

    print("\nPunctul D)")
    with pm.Model() as model_strong:
        # avem un prior fff mic ceea ce ne da un interval mult mai mic de certitudine
        m = pm.Normal('m', mu=50, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=10)
        #likelihood ul
        y = pm.Normal('y', mu=m, sigma=sigma, observed=data)
        strong = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=74, cores=1)

    summary_strong = az.summary(strong, hdi_prob=0.95)
    print(summary_strong)

    mean_strong = strong.posterior['m'].mean().values
    std_strong = strong.posterior['sigma'].mean().values

    print("-" * 40)
    print(f"{'Frequentist':<20} {mean_freq:<10.2f} {std_freq:<10.2f}")
    print(f"{'bayesian (weak)':<20} {mean_bayes:<10.2f} {std_bayes:<10.2f}")
    print(f"{'bayesian (strong)':<20} {mean_strong:<10.2f} {std_strong:<10.2f}")



    # vizualizarea plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extragem samples
    mu_weak = weak.posterior['m'].values.flatten()
    sigma_weak = weak.posterior['sigma'].values.flatten()
    mu_strong = strong.posterior['m'].values.flatten()
    sigma_strong = strong.posterior['sigma'].values.flatten()

    # Plot 1: Posterior μ (weak prior)
    axes[0,0].hist(mu_weak, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(mean_freq, color='red', linestyle='--', linewidth=2, label='Frequentist')
    axes[0,0].axvline(np.mean(mu_weak), color='blue', linestyle='-', linewidth=2, label='Bayesian mean')
    axes[0,0].set_xlabel('μ')
    axes[0,0].set_ylabel('Densitate')
    axes[0,0].set_title('Posterior μ (Prior Slab)')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    # Plot 2: Posterior σ (weak prior)
    axes[0,1].hist(sigma_weak, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].axvline(std_freq, color='red', linestyle='--', linewidth=2, label='Frequentist')
    axes[0,1].axvline(np.mean(sigma_weak), color='darkred', linestyle='-', linewidth=2, label='Bayesian mean')
    axes[0,1].set_xlabel('σ')
    axes[0,1].set_ylabel('Densitate')
    axes[0,1].set_title('Posterior σ (Prior Slab)')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    # Plot 3: Posterior μ (strong prior)
    axes[1,0].hist(mu_strong, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,0].axvline(mean_freq, color='red', linestyle='--', linewidth=2, label='Frequentist')
    axes[1,0].axvline(50, color='green', linestyle=':', linewidth=2, label='Prior mean (50)')
    axes[1,0].axvline(np.mean(mu_strong), color='darkgreen', linestyle='-', linewidth=2, label='Bayesian mean')
    axes[1,0].set_xlabel('μ')
    axes[1,0].set_ylabel('Densitate')
    axes[1,0].set_title('Posterior μ (Prior Puternic)')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # Plot 4: Comparație distribuții μ
    axes[1,1].hist(mu_weak, bins=50, alpha=0.5, label='Weak prior', density=True, color='blue')
    axes[1,1].hist(mu_strong, bins=50, alpha=0.5, label='Strong prior', density=True, color='green')
    axes[1,1].axvline(mean_freq, color='red', linestyle='--', linewidth=2, label='Frequentist')
    axes[1,1].set_xlabel('μ')
    axes[1,1].set_ylabel('Densitate')
    axes[1,1].set_title('Comparație Posterioare μ')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('bayesian_inference.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_bayesian_analysis()