import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy import stats

np.random.seed(74)
#setari pt grafice
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


#incarcam datasetul
df = pd.read_csv('./bike_daily.csv')

print(f"\nDataset shape: {df.shape}")
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
print(f"\nSeason distribution:")
print(df['season'].value_counts())
print(f"\nHoliday distribution:")
print(df['is_holiday'].value_counts())


# exploram relatiile dintre variabile
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# temp vs rental
axes[0, 0].scatter(df['temp_c'], df['rentals'], alpha=0.6, c='blue')
axes[0, 0].set_xlabel('temperatura', fontsize=12)
axes[0, 0].set_ylabel('nr rental', fontsize=12)
axes[0, 0].set_title('temperatura vs rental', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# umiditate vs rental
axes[0, 1].scatter(df['humidity'], df['rentals'], alpha=0.6, c='green')
axes[0, 1].set_xlabel('umiditatea', fontsize=12)
axes[0, 1].set_ylabel('nr rental', fontsize=12)
axes[0, 1].set_title('umiditate vs rental', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# wind speed vs rental
axes[1, 0].scatter(df['wind_kph'], df['rentals'], alpha=0.6, c='red')
axes[1, 0].set_xlabel('wiind speed km/h', fontsize=12)
axes[1, 0].set_ylabel('nr rental', fontsize=12)
axes[1, 0].set_title('wind speed vs rentals', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# boxplot pe sezoane
df.boxplot(column='rentals', by='season', ax=axes[1, 1])
axes[1, 1].set_xlabel('Sezonul', fontsize=12)
axes[1, 1].set_ylabel('nr rentals', fontsize=12)
axes[1, 1].set_title('Inchirieri pe sez', fontsize=14, fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.savefig('./ex1_exploratory_analysis.png', dpi=300, bbox_inches='tight')

numeric_cols = ['rentals', 'temp_c', 'humidity', 'wind_kph']
correlation_matrix = df[numeric_cols].corr()

print(f"\nCorrelation:")
print(correlation_matrix)
# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Continuous Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./ex1_heatmap.png', dpi=300, bbox_inches='tight')

