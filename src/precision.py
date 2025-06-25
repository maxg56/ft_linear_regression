#!/usr/bin/env python3
"""
Precision calculation program for ft_linear_regression project.
Calculates and displays various accuracy metrics for the linear regression algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils.load_csv import load_csv
from const import THETA0, THETA1, MEAN_KM, STD_KM, MEAN_PRICE, STD_PRICE

def calculate_predictions(mileage):
    """
    Calculate predictions using the trained model parameters.
    
    :param mileage: Array of mileage values
    :return: Array of predicted prices
    """
    predictions = []
    for km in mileage:
        # Normalize input
        km_normalized = (km - MEAN_KM) / STD_KM
        # Apply model
        price_normalized = THETA0 + (THETA1 * km_normalized)
        # Denormalize output
        pred_price = price_normalized * STD_PRICE + MEAN_PRICE
        predictions.append(pred_price)
    
    return np.array(predictions)

def calculate_precision_metrics():
    """
    Calculate comprehensive precision metrics for the linear regression model.
    """
    print("=== Calcul de la Précision de l'Algorithme ===")
    
    # Load data from CSV file
    data = load_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return None
    
    # Extract features and target
    mileage = data['km'].values
    actual_prices = data['price'].values
    
    # Calculate predictions
    predicted_prices = calculate_predictions(mileage)
    
    # Calculate various precision metrics
    
    # 1. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    
    # 2. Mean Squared Error (MSE)
    mse = np.mean((actual_prices - predicted_prices) ** 2)
    
    # 3. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # 4. Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    
    # 5. R-squared (Coefficient of Determination)
    ss_res = np.sum((actual_prices - predicted_prices) ** 2)
    ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 6. Adjusted R-squared
    n = len(actual_prices)  # number of observations
    p = 1  # number of predictors (just mileage)
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    
    # 7. Maximum and minimum errors
    errors = actual_prices - predicted_prices
    max_error = np.max(np.abs(errors))
    max_error_idx = np.argmax(np.abs(errors))
    min_error = np.min(np.abs(errors))
    min_error_idx = np.argmin(np.abs(errors))
    
    # 8. Standard deviation of errors
    std_error = np.std(errors)
    
    # 9. Correlation coefficient
    correlation = np.corrcoef(mileage, actual_prices)[0, 1]
    
    # 10. Percentage of predictions within certain thresholds
    within_500 = np.sum(np.abs(errors) <= 500) / len(errors) * 100
    within_1000 = np.sum(np.abs(errors) <= 1000) / len(errors) * 100
    within_1500 = np.sum(np.abs(errors) <= 1500) / len(errors) * 100
    
    # Display results
    print(f"\n📊 MÉTRIQUES DE PRÉCISION")
    print("=" * 50)
    print(f"Nombre de points de données: {n}")
    print(f"Plage de prix: {np.min(actual_prices):.0f} € - {np.max(actual_prices):.0f} €")
    print(f"Prix moyen: {np.mean(actual_prices):.2f} €")
    print()
    
    print("🎯 ERREURS MOYENNES:")
    print(f"   Mean Absolute Error (MAE):           {mae:.2f} €")
    print(f"   Mean Squared Error (MSE):            {mse:.2f} €²")
    print(f"   Root Mean Squared Error (RMSE):      {rmse:.2f} €")
    print(f"   Mean Absolute Percentage Error:      {mape:.2f} %")
    print(f"   Écart-type des erreurs:              {std_error:.2f} €")
    print()
    
    print("📈 COEFFICIENTS DE DÉTERMINATION:")
    print(f"   R-squared (R²):                      {r_squared:.4f} ({r_squared*100:.2f}%)")
    print(f"   R-squared ajusté:                    {adj_r_squared:.4f} ({adj_r_squared*100:.2f}%)")
    print(f"   Corrélation (Pearson):               {correlation:.4f}")
    print()
    
    print("🔍 ANALYSE DES ERREURS:")
    print(f"   Erreur maximale:                     {max_error:.2f} € (point {max_error_idx+1})")
    print(f"   Erreur minimale:                     {min_error:.2f} € (point {min_error_idx+1})")
    print()
    
    print("✅ PRÉCISION PAR SEUILS:")
    print(f"   Prédictions à ±500€:                 {within_500:.1f}% ({int(within_500*n/100)}/{n} points)")
    print(f"   Prédictions à ±1000€:                {within_1000:.1f}% ({int(within_1000*n/100)}/{n} points)")
    print(f"   Prédictions à ±1500€:                {within_1500:.1f}% ({int(within_1500*n/100)}/{n} points)")
    print()
    
    # Interpretation
    print("📋 INTERPRÉTATION:")
    if r_squared >= 0.8:
        print("   🟢 Excellente qualité de régression (R² ≥ 0.8)")
    elif r_squared >= 0.6:
        print("   🟡 Bonne qualité de régression (0.6 ≤ R² < 0.8)")
    elif r_squared >= 0.4:
        print("   🟠 Qualité modérée (0.4 ≤ R² < 0.6)")
    else:
        print("   🔴 Qualité faible (R² < 0.4)")
    
    avg_price = np.mean(actual_prices)
    if mae <= avg_price * 0.1:
        print("   🟢 Erreur absolue excellente (≤ 10% du prix moyen)")
    elif mae <= avg_price * 0.2:
        print("   🟡 Erreur absolue acceptable (≤ 20% du prix moyen)")
    else:
        print("   🔴 Erreur absolue élevée (> 20% du prix moyen)")
    
    if mape <= 10:
        print("   🟢 Erreur pourcentage excellente (≤ 10%)")
    elif mape <= 20:
        print("   🟡 Erreur pourcentage acceptable (≤ 20%)")
    else:
        print("   🔴 Erreur pourcentage élevée (> 20%)")
    
    print()
    
    # Return metrics for potential further use
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'correlation': correlation,
        'max_error': max_error,
        'min_error': min_error,
        'std_error': std_error,
        'within_500': within_500,
        'within_1000': within_1000,
        'within_1500': within_1500,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'errors': errors
    }
    
    return metrics

def plot_precision_analysis(metrics):
    """
    Create visualizations for precision analysis.
    
    :param metrics: Dictionary containing calculated metrics
    """
    if metrics is None:
        return
    
    actual_prices = metrics['actual_prices']
    predicted_prices = metrics['predicted_prices']
    errors = metrics['errors']
    
    # Create comprehensive precision plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(actual_prices, predicted_prices, alpha=0.7, color='blue', s=60)
    
    # Perfect prediction line
    min_price = min(np.min(actual_prices), np.min(predicted_prices))
    max_price = max(np.max(actual_prices), np.max(predicted_prices))
    axes[0, 0].plot([min_price, max_price], [min_price, max_price], 
                    'r--', linewidth=2, label='Prédiction parfaite')
    
    axes[0, 0].set_xlabel('Prix Réel (€)')
    axes[0, 0].set_ylabel('Prix Prédit (€)')
    axes[0, 0].set_title('Prix Prédits vs Prix Réels')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² to the plot
    axes[0, 0].text(0.05, 0.95, f'R² = {metrics["r_squared"]:.4f}', 
                    transform=axes[0, 0].transAxes, fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Residuals plot
    axes[0, 1].scatter(predicted_prices, errors, alpha=0.7, color='green', s=60)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prix Prédit (€)')
    axes[0, 1].set_ylabel('Résidus (€)')
    axes[0, 1].set_title('Analyse des Résidus')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution histogram
    axes[1, 0].hist(errors, bins=12, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Erreurs (€)')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('Distribution des Erreurs')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add statistics to histogram
    stats_text = f'MAE: {metrics["mae"]:.0f}€\nRMSE: {metrics["rmse"]:.0f}€\nMAPE: {metrics["mape"]:.1f}%'
    axes[1, 0].text(0.70, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 4. Absolute errors plot
    abs_errors = np.abs(errors)
    axes[1, 1].bar(range(len(abs_errors)), abs_errors, alpha=0.7, color='purple')
    axes[1, 1].axhline(y=metrics['mae'], color='red', linestyle='--', 
                       linewidth=2, label=f'MAE = {metrics["mae"]:.0f}€')
    axes[1, 1].set_xlabel('Index des Points de Données')
    axes[1, 1].set_ylabel('Erreur Absolue (€)')
    axes[1, 1].set_title('Erreurs Absolues par Point')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../graphs/precision_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique d'analyse de précision sauvegardé: ../graphs/precision_analysis.png")

def detailed_error_analysis():
    """
    Provide detailed analysis of individual prediction errors.
    """
    print("\n🔍 ANALYSE DÉTAILLÉE DES ERREURS")
    print("=" * 60)
    
    # Load data
    data = load_csv("../data/data.csv")
    if data is None or len(data) == 0:
        return
    
    mileage = data['km'].values
    actual_prices = data['price'].values
    predicted_prices = calculate_predictions(mileage)
    errors = actual_prices - predicted_prices
    
    # Create detailed table
    print(f"{'#':>3} {'Kilométrage':>12} {'Prix Réel':>10} {'Prix Prédit':>12} {'Erreur':>8} {'Erreur %':>9}")
    print("-" * 60)
    
    for i, (km, actual, predicted, error) in enumerate(zip(mileage, actual_prices, predicted_prices, errors)):
        error_pct = (error / actual) * 100
        print(f"{i+1:>3} {km:>12.0f} {actual:>10.0f} {predicted:>12.2f} {error:>8.0f} {error_pct:>8.1f}%")
    
    print("-" * 60)
    print(f"Erreur absolue moyenne: {np.mean(np.abs(errors)):.2f} €")
    print(f"Erreur pourcentage moyenne: {np.mean(np.abs(errors/actual_prices)*100):.2f} %")

def main():
    """
    Main function to calculate and display precision metrics.
    """
    print("🎯 PROGRAMME DE CALCUL DE PRÉCISION")
    print("=" * 50)
    
    try:
        # Calculate precision metrics
        metrics = calculate_precision_metrics()
        
        if metrics is not None:
            # Generate precision analysis plots
            print("📊 Génération des graphiques d'analyse de précision...")
            plot_precision_analysis(metrics)
            
            # Detailed error analysis
            detailed_error_analysis()
            
            print("\n✅ Analyse de précision terminée avec succès!")
        
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("Assurez-vous que toutes les dépendances sont installées.")
    except Exception as e:
        print(f"Erreur lors du calcul de précision: {e}")

if __name__ == "__main__":
    main()
