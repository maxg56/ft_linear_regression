#!/usr/bin/env python3
"""
Data visualization program for ft_linear_regression project.
Plots the data distribution and the resulting linear regression line.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from utils.load_csv import load_csv
from const import THETA0, THETA1, MEAN_KM, STD_KM, MEAN_PRICE, STD_PRICE, DATA_FILE

def plot_data_distribution():
    """
    Plot the data distribution to see how the data points are spread.
    """
    # Load data from CSV file
    data = load_csv(DATA_FILE)
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return
    
    # Extract features and target
    mileage = data['km'].values
    price = data['price'].values
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Main scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(mileage, price, color='blue', alpha=0.7, s=60, edgecolors='darkblue')
    plt.xlabel('Kilométrage (km)', fontsize=11)
    plt.ylabel('Prix (€)', fontsize=11)
    plt.title('Distribution des Données - Prix vs Kilométrage', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Nombre de points: {len(mileage)}\n'
    stats_text += f'Kilométrage moyen: {np.mean(mileage):.0f} km\n'
    stats_text += f'Prix moyen: {np.mean(price):.0f} €\n'
    stats_text += f'Étendue km: {np.min(mileage):.0f} - {np.max(mileage):.0f}\n'
    stats_text += f'Étendue prix: {np.min(price):.0f} - {np.max(price):.0f} €'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Histogram of mileage
    plt.subplot(2, 2, 2)
    plt.hist(mileage, bins=10, color='skyblue', alpha=0.7, edgecolor='darkblue')
    plt.xlabel('Kilométrage (km)', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title('Distribution du Kilométrage', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Histogram of prices
    plt.subplot(2, 2, 3)
    plt.hist(price, bins=10, color='lightcoral', alpha=0.7, edgecolor='darkred')
    plt.xlabel('Prix (€)', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title('Distribution des Prix', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Box plot for outlier detection
    plt.subplot(2, 2, 4)
    box_data = [mileage/1000, price/1000]  # Scale for better visualization
    plt.boxplot(box_data, labels=['Kilométrage (×1000)', 'Prix (×1000)'])
    plt.title('Détection des Valeurs Aberrantes', fontsize=12, fontweight='bold')
    plt.ylabel('Valeurs (×1000)', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('../graphs/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique de distribution sauvegardé: ../graphs/data_distribution.png")

def plot_regression_line():
    """
    Plot the data points with the linear regression line.
    This shows the result of the linear regression algorithm.
    """
    # Load data from CSV file
    data = load_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return
    
    # Extract features and target
    mileage = data['km'].values
    price = data['price'].values
    
    # Generate predictions using trained model parameters
    predictions = []
    for km in mileage:
        # Normalize input
        km_normalized = (km - MEAN_KM) / STD_KM
        # Apply model
        price_normalized = THETA0 + (THETA1 * km_normalized)
        # Denormalize output
        pred_price = price_normalized * STD_PRICE + MEAN_PRICE
        predictions.append(pred_price)
    
    predictions = np.array(predictions)
    
    # Create a smooth line for visualization
    km_range = np.linspace(np.min(mileage), np.max(mileage), 100)
    smooth_predictions = []
    for km in km_range:
        km_normalized = (km - MEAN_KM) / STD_KM
        price_normalized = THETA0 + (THETA1 * km_normalized)
        pred_price = price_normalized * STD_PRICE + MEAN_PRICE
        smooth_predictions.append(pred_price)
    
    smooth_predictions = np.array(smooth_predictions)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Scatter plot of actual data
    plt.scatter(mileage, price, color='blue', alpha=0.7, s=60, 
               label='Données réelles', edgecolors='darkblue', linewidth=0.5)
    
    # Plot regression line
    plt.plot(km_range, smooth_predictions, color='red', linewidth=3, 
             label='Ligne de régression linéaire', alpha=0.8)
    
    # Plot predictions for actual data points
    plt.scatter(mileage, predictions, color='orange', alpha=0.6, s=40,
               label='Prédictions du modèle', marker='x', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Kilométrage (km)', fontsize=12)
    plt.ylabel('Prix (€)', fontsize=12)
    plt.title('Résultat de la Régression Linéaire\nPrix des Voitures vs Kilométrage', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add equation and parameters
    equation_text = f'Équation: y = θ₀ + θ₁ × x_norm\n'
    equation_text += f'θ₀ (intercept): {THETA0:.6f}\n'
    equation_text += f'θ₁ (slope): {THETA1:.6f}\n\n'
    equation_text += f'Normalisation:\n'
    equation_text += f'μ_km = {MEAN_KM:.0f}, σ_km = {STD_KM:.0f}\n'
    equation_text += f'μ_prix = {MEAN_PRICE:.0f}, σ_prix = {STD_PRICE:.0f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, equation_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Calculate and display correlation
    correlation = np.corrcoef(mileage, price)[0, 1]
    corr_text = f'Corrélation: {correlation:.4f}'
    plt.text(0.02, 0.02, corr_text, transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../graphs/regression_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique de régression sauvegardé: ../graphs/regression_result.png")

def main():
    """
    Main function to generate both plots.
    """
    print("=== Visualisation des Données et Régression Linéaire ===")
    
    try:
        print("\n1. Génération du graphique de distribution des données...")
        plot_data_distribution()
        
        print("\n2. Génération du graphique avec la ligne de régression...")
        plot_regression_line()
        
        print("\n✅ Graphiques générés avec succès !")
        print("   - Distribution des données: ../graphs/data_distribution.png")
        print("   - Ligne de régression: ../graphs/regression_result.png")
        
    except ImportError:
        print("Erreur: matplotlib n'est pas installé.")
        print("Installez-le avec: pip install matplotlib")
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques: {e}")

if __name__ == "__main__":
    main()
