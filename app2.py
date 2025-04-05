# -*- coding: utf-8 -*-
# """
# Devoir: Réduction de dimension par ACM sur le dataset Adult
# """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prince import MCA
from sklearn.preprocessing import LabelEncoder

# 1. Chargement et préparation des données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 
           'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
           'native-country', 'income']

df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df = df.dropna()

# Sélection et catégorisation des variables qualitatives
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country', 'income']
df_cat = df[cat_cols]

# 2. Application de l'ACM
mca = MCA(n_components=10, n_iter=3, random_state=42)
mca.fit(df_cat)

# 3. Analyse des résultats
# Variance expliquée
eigenvalues = mca.eigenvalues_
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

print("\nVariance expliquée par chaque axe:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"Axe {i+1}: {var:.3f} ({cum_var:.3f} cumulée)")

# 4. Visualisation
plt.figure(figsize=(15, 6))

# Eboulis des valeurs propres
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5, align='center')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid')
plt.xlabel('Axe factoriel')
plt.ylabel('Variance expliquée')
plt.title('Eboulis des valeurs propres')

# Carte factorielle
plt.subplot(1, 2, 2)
row_coords = mca.row_coordinates(df_cat)
plt.scatter(row_coords[0], row_coords[1], alpha=0.5)
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title('Carte factorielle des modalités (2 premiers axes)')
plt.title('Carte factorielle des modalités (2 premiers axes)')
plt.tight_layout()
plt.show()

# 5. Contribution des modalités aux axes
contributions = mca.column_contributions_
print("\nContributions des modalités aux 2 premiers axes:")
print(contributions.head(10))

# 6. Représentation des individus avec une variable illustrative (income)
plt.figure(figsize=(10, 8))
colors = {'<=50K': 'blue', '>50K': 'red'}
for income, color in colors.items():
    idx = df_cat['income'] == income
    plt.scatter(mca.row_coordinates(df_cat)[idx][0], 
                mca.row_coordinates(df_cat)[idx][1], 
                c=color, label=income, alpha=0.3)
plt.legend()
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title('Projection des individus colorés par revenu')
plt.show()

# 7. Interprétation qualitative
# Modalités les plus contributives
top_modalites_axe1 = contributions[0].sort_values(ascending=False).head(5)
top_modalites_axe2 = contributions[1].sort_values(ascending=False).head(5)

print("\nModalités les plus contributives à l'axe 1:")
print(top_modalites_axe1)
print("\nModalités les plus contributives à l'axe 2:")
print(top_modalites_axe2)