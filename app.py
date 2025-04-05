# devoir_acm.py
# Analyse des Correspondances Multiples (ACM) sur des données catégorielles
# Dataset utilisé : Titanic (de Seaborn)
# Étudiant : [Votre Nom]

# 1. Import des bibliothèques
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import prince
import numpy as np

# 2. Chargement du dataset Titanic
df = sns.load_dataset('titanic')

# 3. Sélection des variables catégorielles
categorical_df = df[['sex', 'class', 'embark_town', 'who', 'deck', 'alone']]

# 4. Nettoyage des données (suppression des lignes avec des valeurs manquantes)
categorical_df = categorical_df.dropna()

# 5. Affichage des premières lignes pour vérifier
print("Données catégorielles sélectionnées :")
print(categorical_df.head())

# 6. Application de l'ACM avec prince
# On garde 2 composantes pour la visualisation
mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(categorical_df)
mca_result = mca.transform(categorical_df)

# 7. Ajout des composantes au DataFrame original
categorical_df['ACM_1'] = mca_result[0]
categorical_df['ACM_2'] = mca_result[1]

# 8. Visualisation des individus dans l'espace réduit
plt.figure(figsize=(10, 6))
sns.scatterplot(data=categorical_df, x='ACM_1', y='ACM_2', hue='class', palette='Set2')
plt.title("Projection des individus avec ACM (Titanic Dataset)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.legend(title="Classe")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Affichage de la variance expliquée par les composantes
explained_inertia = mca.explained_inertia_
print("\nVariance expliquée par les composantes :")
for i, var in enumerate(explained_inertia):
    print(f"Composante {i+1}: {var*100:.2f}%")

# 10. Graphique de la variance cumulée
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_inertia), marker='o')
plt.title("Variance expliquée cumulée par les composantes (ACM)")
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.grid(True)
plt.tight_layout()
plt.show()
