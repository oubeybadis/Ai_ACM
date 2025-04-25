# oubey badis G2
# 2025-04-26

# data source : https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Télécharger et charger les données depuis Kaggle
# Nous allons utiliser les fichiers orders et order_items
orders = pd.read_csv('olist_orders_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')

# Fusionner les données pour obtenir les transactions
transactions = order_items.merge(products, on='product_id', how='left')
transactions = transactions.merge(orders, on='order_id', how='left')

# Créer un DataFrame avec une ligne par transaction et une colonne par catégorie de produit
# Nous allons utiliser la catégorie de produit comme "item" pour nos règles d'association
basket = transactions.groupby(['order_id', 'product_category_name']).size().unstack().reset_index().fillna(0)

# Convertir les quantités en 1 (présence) ou 0 (absence)
basket_sets = basket.drop('order_id', axis=1)
basket_sets = basket_sets.applymap(lambda x: 1 if x > 0 else 0)

# Afficher un aperçu des données préparées
print("Aperçu des données préparées pour l'extraction des règles d'association:")
print(basket_sets.head())
print(f"Nombre de transactions: {len(basket_sets)}")
print(f"Nombre de catégories de produits: {len(basket_sets.columns)}")

# Sauvegarder les données préparées
basket_sets.to_csv('olist_transactions.csv', index=False)

def run_apriori(data, min_support, min_confidence, min_lift):
    start_time = time.time()
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    end_time = time.time()
    execution_time = end_time - start_time
    return rules, execution_time, frequent_itemsets

def run_fpgrowth(data, min_support, min_confidence, min_lift):
    start_time = time.time()
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    end_time = time.time()
    execution_time = end_time - start_time
    return rules, execution_time, frequent_itemsets

# Test de l'algorithme Apriori avec différents seuils de support
min_supports = [0.01, 0.02, 0.03, 0.04, 0.05]
apriori_results = []
fpgrowth_results = []

for min_support in min_supports:
    rules, execution_time, frequent_itemsets = run_apriori(basket_sets, min_support=min_support, min_confidence=0.3, min_lift=1.0)
    apriori_results.append({
        'min_support': min_support,
        'execution_time': execution_time,
        'nb_rules': len(rules),
        'nb_frequent_itemsets': len(frequent_itemsets)
    })
    print(f"Apriori - Support: {min_support}, Temps: {execution_time:.2f}s, Règles: {len(rules)}, Itemsets fréquents: {len(frequent_itemsets)}")

    rules, execution_time, frequent_itemsets = run_fpgrowth(basket_sets, min_support=min_support, min_confidence=0.3, min_lift=1.0)
    fpgrowth_results.append({
        'min_support': min_support,
        'execution_time': execution_time,
        'nb_rules': len(rules),
        'nb_frequent_itemsets': len(frequent_itemsets)
    })
    print(f"FP-Growth - Support: {min_support}, Temps: {execution_time:.2f}s, Règles: {len(rules)}, Itemsets fréquents: {len(frequent_itemsets)}")

# Convertir les résultats en DataFrame pour faciliter l'analyse
apriori_df = pd.DataFrame(apriori_results)
fpgrowth_df = pd.DataFrame(fpgrowth_results)

# Comparaison du temps d'exécution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(apriori_df['min_support'], apriori_df['execution_time'], marker='o', label='Apriori')
plt.plot(fpgrowth_df['min_support'], fpgrowth_df['execution_time'], marker='s', label='FP-Growth')
plt.title('Temps d\'exécution vs Support Minimum')
plt.xlabel('Support Minimum')
plt.ylabel('Temps d\'exécution (s)')
plt.legend()
plt.grid(True)

# Comparaison du nombre de règles générées
plt.subplot(1, 2, 2)
plt.plot(apriori_df['min_support'], apriori_df['nb_rules'], marker='o', label='Apriori')
plt.plot(fpgrowth_df['min_support'], fpgrowth_df['nb_rules'], marker='s', label='FP-Growth')
plt.title('Nombre de règles vs Support Minimum')
plt.xlabel('Support Minimum')
plt.ylabel('Nombre de règles')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_algorithms.png')
plt.show()

# Analyse des règles d'association obtenues
# Pour un support de 0.02 par exemple
apriori_rules, _, _ = run_apriori(basket_sets, min_support=0.02, min_confidence=0.3, min_lift=1.0)
fpgrowth_rules, _, _ = run_fpgrowth(basket_sets, min_support=0.02, min_confidence=0.3, min_lift=1.0)

# Afficher les 10 premières règles pour chaque algorithme
print("\nLes 10 premières règles avec Apriori:")
print(apriori_rules.sort_values('lift', ascending=False).head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\nLes 10 premières règles avec FP-Growth:")
print(fpgrowth_rules.sort_values('lift', ascending=False).head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Comparer l'impact du seuil de confiance
min_confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
confidence_results = []

for min_confidence in min_confidences:
    apriori_rules, apriori_time, _ = run_apriori(basket_sets, min_support=0.02, min_confidence=min_confidence, min_lift=1.0)
    fpgrowth_rules, fpgrowth_time, _ = run_fpgrowth(basket_sets, min_support=0.02, min_confidence=min_confidence, min_lift=1.0)
    
    confidence_results.append({
        'min_confidence': min_confidence,
        'apriori_rules': len(apriori_rules),
        'fpgrowth_rules': len(fpgrowth_rules),
        'apriori_time': apriori_time,
        'fpgrowth_time': fpgrowth_time
    })

confidence_df = pd.DataFrame(confidence_results)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(confidence_df['min_confidence'], confidence_df['apriori_rules'], marker='o', label='Apriori')
plt.plot(confidence_df['min_confidence'], confidence_df['fpgrowth_rules'], marker='s', label='FP-Growth')
plt.title('Nombre de règles vs Confiance Minimum')
plt.xlabel('Confiance Minimum')
plt.ylabel('Nombre de règles')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(confidence_df['min_confidence'], confidence_df['apriori_time'], marker='o', label='Apriori')
plt.plot(confidence_df['min_confidence'], confidence_df['fpgrowth_time'], marker='s', label='FP-Growth')
plt.title('Temps d\'exécution vs Confiance Minimum')
plt.xlabel('Confiance Minimum')
plt.ylabel('Temps d\'exécution (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_confidence.png')
plt.show()

# Fonction pour visualiser les règles d'association sous forme de réseau
def visualize_rules(rules, title):
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Créer un graphe dirigé
    G = nx.DiGraph()
    
    # Ajouter les arêtes (règles d'association)
    for _, row in rules.head(20).iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # Convertir frozenset en chaîne de caractères pour l'affichage
        for antecedent in antecedents:
            for consequent in consequents:
                G.add_edge(antecedent, consequent, 
                           weight=row['lift'], 
                           support=row['support'],
                           confidence=row['confidence'])
    
    # Préparer le layout
    pos = nx.spring_layout(G, k=0.5)
    
    # Dessiner les nœuds
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
    
    # Dessiner les arêtes avec couleur basée sur le lift
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color=weights, edge_cmap=plt.cm.Reds)
    
    # Ajouter les labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Visualiser les règles d'association pour les deux algorithmes
best_apriori_rules, _, _ = run_apriori(basket_sets, min_support=0.02, min_confidence=0.3, min_lift=2.0)
best_fpgrowth_rules, _, _ = run_fpgrowth(basket_sets, min_support=0.02, min_confidence=0.3, min_lift=2.0)

visualize_rules(best_apriori_rules, "Règles d'association avec Apriori")
visualize_rules(best_fpgrowth_rules, "Règles d'association avec FP-Growth")