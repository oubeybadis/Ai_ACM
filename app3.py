# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA  # Bibliothèque pour l'ACM (Analyse des Correspondances Multiples)
from sklearn.preprocessing import KBinsDiscretizer
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Fonction pour créer des ellipses de confiance
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, 
                      facecolor=facecolor, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Chargement des données
print("Chargement des données Wine Quality...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# 1. ANALYSE EXPLORATOIRE DES DONNÉES
print("\n1. ANALYSE EXPLORATOIRE DES DONNÉES\n" + "="*40)
print(f"Dimensions du jeu de données: {data.shape[0]} lignes, {data.shape[1]} colonnes")
print("\nAperçu des 5 premières lignes:")
print(data.head())

print("\nStatistiques descriptives:")
print(data.describe().round(2))

print("\nVérification des valeurs manquantes:")
missing_values = data.isnull().sum()
print(missing_values)

# Distribution de la qualité (variable cible)
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=data)
plt.title('Distribution de la qualité des vins')
plt.xlabel('Niveau de qualité')
plt.ylabel('Nombre de vins')
plt.grid(axis='y', alpha=0.3)
plt.savefig('quality_distribution.png')

# 2. TRANSFORMATION DES DONNÉES POUR L'ACM
print("\n2. TRANSFORMATION DES DONNÉES QUANTITATIVES EN CATÉGORIELLES\n" + "="*40)

# Création d'une copie des données pour la discrétisation
data_cat = data.copy()

# Discrétisation des variables quantitatives en catégories
# Nous utilisons KBinsDiscretizer pour créer des bins équilibrés
n_bins = 3  # Nombre de catégories pour chaque variable (faible, moyen, élevé)

# Liste des colonnes à discrétiser (toutes sauf quality)
cols_to_discretize = [col for col in data.columns if col != 'quality']

# Définition des labels pour les catégories
labels = ['faible', 'moyen', 'eleve']

# Discrétisation de chaque variable
for col in cols_to_discretize:
    # Utilisation de KBinsDiscretizer avec stratégie quantile
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    data_cat[col] = discretizer.fit_transform(data[col].values.reshape(-1, 1))
    
    # Conversion des codes numériques en labels descriptifs
    data_cat[col] = data_cat[col].map({0: f"{col}_faible", 
                                       1: f"{col}_moyen", 
                                       2: f"{col}_eleve"})

# Conversion de la qualité en catégories
# Regroupement des qualités en 3 catégories: faible (3-4), moyenne (5-6), élevée (7-8)
data_cat['quality'] = pd.cut(data['quality'], 
                            bins=[2, 4, 6, 9], 
                            labels=['qualite_faible', 'qualite_moyenne', 'qualite_elevee'])

print("\nAperçu des données catégorisées:")
print(data_cat.head())

# Tableau de contingence entre qualité et acidité volatile
contingency_table = pd.crosstab(data_cat['quality'], data_cat['volatile acidity'])
print("\nTableau de contingence entre qualité et acidité volatile:")
print(contingency_table)

# Test de chi2 pour vérifier les associations entre les variables
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nTest Chi² entre qualité et acidité volatile: chi²={chi2:.2f}, p-value={p:.4f}")

# 3. APPLICATION DE L'ACM
print("\n3. APPLICATION DE L'ANALYSE DES CORRESPONDANCES MULTIPLES\n" + "="*40)

# On convertit d'abord le dataframe en format disjonctif complet
# En utilisant la fonction get_dummies de pandas
data_dummies = pd.get_dummies(data_cat)

# Application de l'ACM avec la bibliothèque prince
mca = MCA(n_components=5, n_iter=3, random_state=42)
mca = mca.fit(data_dummies)

# Extraction des coordonnées des individus
individuals = mca.row_coordinates(data_dummies)

# Extraction des coordonnées des modalités (variables)
variables = mca.column_coordinates(data_dummies)

# Variance expliquée par les axes
eigenvalues = mca.eigenvalues_
print("\nValeurs propres (eigenvalues):")
print(eigenvalues)

explained_inertia = eigenvalues / sum(eigenvalues) * 100
cumulative_inertia = np.cumsum(explained_inertia)

print("\nInertie expliquée par axe (%):")
for i, inertia in enumerate(explained_inertia[:5]):
    print(f"Axe {i+1}: {inertia:.2f}% ({cumulative_inertia[i]:.2f}% cumulée)")

# 4. VISUALISATION DES RÉSULTATS
print("\n4. VISUALISATION DES RÉSULTATS DE L'ACM\n" + "="*40)

# Nuage des individus
plt.figure(figsize=(12, 10))
scatter = plt.scatter(individuals[0], individuals[1], 
                    c=data['quality'], cmap='viridis', 
                    alpha=0.6, s=50, edgecolors='w')
plt.colorbar(scatter, label='Qualité du vin (originale)')

# Ajout d'ellipses pour les différentes qualités
quality_categories = data_cat['quality'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(quality_categories)))

for quality, color in zip(quality_categories, colors):
    mask = data_cat['quality'] == quality
    if sum(mask) > 2:  # Au moins 3 points pour calculer une ellipse
        confidence_ellipse(individuals.loc[mask, 0], individuals.loc[mask, 1], 
                         plt.gca(), n_std=2.0, edgecolor=color, alpha=0.3, 
                         label=quality)

plt.title('Projection des vins sur les deux premiers axes factoriels', fontsize=14)
plt.xlabel(f'Axe 1 ({explained_inertia[0]:.2f}% d\'inertie)', fontsize=12)
plt.ylabel(f'Axe 2 ({explained_inertia[1]:.2f}% d\'inertie)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Qualité du vin')
plt.tight_layout()
plt.savefig('acm_individuals.png')

# Nuage des modalités (variables)
plt.figure(figsize=(14, 12))
# Extraction des modalités liées à la qualité pour coloration spécifique
quality_vars = [col for col in variables.index if 'qualite_' in col]
other_vars = [col for col in variables.index if 'qualite_' not in col]

# Tracer les points pour les modalités autres que la qualité
plt.scatter(variables.loc[other_vars, 0], variables.loc[other_vars, 1], 
           c='blue', alpha=0.7, s=60)

# Tracer les points pour les modalités de qualité avec une couleur différente
plt.scatter(variables.loc[quality_vars, 0], variables.loc[quality_vars, 1], 
           c='red', alpha=0.9, s=80, marker='*')

# Ajout des étiquettes pour les modalités
for i, (idx, row) in enumerate(variables.iterrows()):
    plt.annotate(idx, (row[0], row[1]), 
                fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title('Projection des modalités sur les deux premiers axes factoriels', fontsize=14)
plt.xlabel(f'Axe 1 ({explained_inertia[0]:.2f}% d\'inertie)', fontsize=12)
plt.ylabel(f'Axe 2 ({explained_inertia[1]:.2f}% d\'inertie)', fontsize=12)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acm_variables.png')

# Visualisation de la variance expliquée (Scree plot)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_inertia) + 1), explained_inertia, alpha=0.8, color='skyblue', label='Inertie individuelle')
plt.step(range(1, len(cumulative_inertia) + 1), cumulative_inertia, where='mid', color='red', label='Inertie cumulée')
plt.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Seuil de 80%')
plt.title('Inertie expliquée par axe factoriel', fontsize=14)
plt.xlabel('Axe factoriel', fontsize=12)
plt.ylabel('Inertie expliquée (%)', fontsize=12)
plt.xticks(range(1, len(explained_inertia) + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('acm_scree_plot.png')

# 5. CONTRIBUTIONS ET COSINUS² DES MODALITÉS
print("\n5. ANALYSE DES CONTRIBUTIONS ET QUALITÉ DE REPRÉSENTATION\n" + "="*40)

# Contributions des modalités aux axes
contributions = mca.column_contributions_
# Debugging: Check the type and shape of contributions
print(f"Type of contributions: {type(contributions)}")
if hasattr(contributions, 'shape'):
    print(f"Shape of contributions: {contributions.shape}")

# Ensure contributions is a NumPy array
contributions = np.array(contributions)


print("\nContributions des modalités aux 2 premiers axes (%):")
contributions_df = pd.DataFrame(contributions[:, :2] * 100, 
                              index=variables.index, 
                              columns=['Axe 1', 'Axe 2'])
print(contributions_df.sort_values(by='Axe 1', ascending=False).head(10))

# Cosinus² des modalités
cos2 = mca.column_cosine_similarities  # Access as an attribute
print(f"Type of cos2: {type(cos2)}")
print(f"Value of cos2: {cos2}")

# Check if cos2 is valid before proceeding
if isinstance(cos2, np.ndarray) and cos2.ndim == 2:
    print("\nCosinus² des modalités sur les 2 premiers axes:")
    cos2_df = pd.DataFrame(cos2[:, :2], 
                           index=variables.index, 
                           columns=['Axe 1', 'Axe 2'])
    print(cos2_df.sort_values(by='Axe 1', ascending=False).head(10))
else:
    print("cos2 is not a valid 2D array. Please check the MCA object or input data.")

# Visualisation des contributions
plt.figure(figsize=(14, 8))
# Sélection des 10 modalités avec les plus fortes contributions sur l'axe 1
top_contrib_axis1 = contributions_df.sort_values(by='Axe 1', ascending=False).head(10)
top_contrib_axis1.plot(kind='bar', width=0.8)
plt.title('Top 10 des modalités contribuant à l\'axe 1', fontsize=14)
plt.xlabel('Modalités', fontsize=12)
plt.ylabel('Contribution (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('acm_contributions_axis1.png')

plt.figure(figsize=(14, 8))
# Sélection des 10 modalités avec les plus fortes contributions sur l'axe 2
top_contrib_axis2 = contributions_df.sort_values(by='Axe 2', ascending=False).head(10)
top_contrib_axis2.plot(kind='bar', width=0.8)
plt.title('Top 10 des modalités contribuant à l\'axe 2', fontsize=14)
plt.xlabel('Modalités', fontsize=12)
plt.ylabel('Contribution (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('acm_contributions_axis2.png')

# 6. ANALYSE DES ASSOCIATIONS ENTRE VARIABLES CATÉGORIELLES
print("\n6. ANALYSE DES ASSOCIATIONS ENTRE VARIABLES CATÉGORIELLES\n" + "="*40)

# Création d'un heatmap des p-values du test Chi²
# Sélection de quelques variables importantes pour la lisibilité
selected_cols = ['quality', 'alcohol', 'volatile acidity', 'sulphates', 'total sulfur dioxide']
chi2_matrix = pd.DataFrame(index=selected_cols, columns=selected_cols)

for col1 in selected_cols:
    for col2 in selected_cols:
        if col1 != col2:
            contingency = pd.crosstab(data_cat[col1], data_cat[col2])
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            chi2_matrix.loc[col1, col2] = -np.log10(p)  # Transformation -log10 pour une meilleure visualisation
        else:
            chi2_matrix.loc[col1, col2] = 0

# Check for missing or invalid values in chi2_matrix
print("\nChi2 Matrix:")
print(chi2_matrix)

# Replace NaN or non-numeric values with 0
chi2_matrix = chi2_matrix.fillna(0)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(chi2_matrix.astype(float), annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Intensité des associations entre variables (-log10(p-value))', fontsize=14)
plt.tight_layout()
plt.savefig('chi2_associations.png')

