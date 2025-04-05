# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA  # Bibliothèque pour l'ACM (Analyse des Correspondances Multiples)
from sklearn.preprocessing import KBinsDiscretizer
import scipy.stats as stats

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Chargement des données Heart Disease
print("Chargement des données Heart Disease...")
# Remplacez le chemin ci-dessous par le chemin local où vous avez enregistré le fichier CSV
file_path = r"c:\Users\USER\Desktop\oubey\M1 IL\S2\Dm\Ai_ACM\heart.csv"
data = pd.read_csv(file_path)

# Vérification des données chargées
print("\nAperçu des 5 premières lignes:")
print(data.head())

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

# Distribution de la variable cible (présence de maladie cardiaque)
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=data)
plt.title('Distribution de la présence de maladie cardiaque')
plt.xlabel('Présence de maladie cardiaque (0 = Non, 1 = Oui)')
plt.ylabel('Nombre de patients')
plt.grid(axis='y', alpha=0.3)
plt.savefig('target_distribution.png')

# 2. TRANSFORMATION DES DONNÉES POUR L'ACM
print("\n2. TRANSFORMATION DES DONNÉES QUANTITATIVES EN CATÉGORIELLES\n" + "="*40)

# Création d'une copie des données pour la discrétisation
data_cat = data.copy()

# Discrétisation des variables quantitatives en catégories
# Nous utilisons KBinsDiscretizer pour créer des bins équilibrés
n_bins = 3  # Nombre de catégories pour chaque variable (faible, moyen, élevé)

# Liste des colonnes à discrétiser (toutes sauf target)
cols_to_discretize = [col for col in data.columns if col != 'target']

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

# Conversion de la cible en catégories
data_cat['target'] = data['target'].map({0: 'Absence', 1: 'Présence'})

print("\nAperçu des données catégorisées:")
print(data_cat.head())

# 3. APPLICATION DE L'ACM
print("\n3. APPLICATION DE L'ANALYSE DES CORRESPONDANCES MULTIPLES\n" + "="*40)

# On convertit d'abord le dataframe en format disjonctif complet
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


# Nuage des individus
plt.figure(figsize=(12, 10))
scatter = plt.scatter(individuals[0], individuals[1], 
                    c=data['target'], cmap='viridis', 
                    alpha=0.6, s=50, edgecolors='w')
plt.colorbar(scatter, label='Présence de maladie cardiaque')

plt.title('Projection des patients sur les deux premiers axes factoriels', fontsize=14)
plt.xlabel(f'Axe 1 ({explained_inertia[0]:.2f}% d\'inertie)', fontsize=12)
plt.ylabel(f'Axe 2 ({explained_inertia[1]:.2f}% d\'inertie)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acm_individuals.png')

# Nuage des modalités (variables)
plt.figure(figsize=(14, 12))
plt.scatter(variables[0], variables[1], c='blue', alpha=0.7, s=60)
for i, (idx, row) in enumerate(variables.iterrows()):
    plt.annotate(idx, (row[0], row[1]), fontsize=9, ha='center', va='center')

plt.title('Projection des modalités sur les deux premiers axes factoriels', fontsize=14)
plt.xlabel(f'Axe 1 ({explained_inertia[0]:.2f}% d\'inertie)', fontsize=12)
plt.ylabel(f'Axe 2 ({explained_inertia[1]:.2f}% d\'inertie)', fontsize=12)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acm_variables.png')