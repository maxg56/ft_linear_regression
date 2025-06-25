#!/usr/bin/env python3
"""
Complete demonstration script for ft_linear_regression project (Refactored Version).
This script demonstrates all the required functionalities:
1. Plotting the data distribution
2. Plotting the linear regression line
3. Calculating the precision of the algorithm
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from utils.load_csv import load_csv
from const import THETA0, THETA1, MEAN_KM, STD_KM, MEAN_PRICE, STD_PRICE

# Convert string constants to float if needed
THETA0 = float(THETA0) if isinstance(THETA0, str) else THETA0
THETA1 = float(THETA1) if isinstance(THETA1, str) else THETA1
MEAN_KM = float(MEAN_KM) if isinstance(MEAN_KM, str) else MEAN_KM
STD_KM = float(STD_KM) if isinstance(STD_KM, str) else STD_KM
MEAN_PRICE = float(MEAN_PRICE) if isinstance(MEAN_PRICE, str) else MEAN_PRICE
STD_PRICE = float(STD_PRICE) if isinstance(STD_PRICE, str) else STD_PRICE


def load_and_prepare_data():
    """
    Load and prepare data for analysis.
    
    Returns:
        tuple: (mileage, actual_prices, predictions) or None if error
    """
    data = load_csv("../data/data.csv")
    if data is None or len(data) == 0:
        print("❌ Erreur: Impossible de charger les données")
        return None
    
    mileage = data['km'].values
    actual_prices = data['price'].values
    
    print(f" Données chargées: {len(mileage)} points")
    print(f"   Kilométrage: {np.min(mileage):.0f} - {np.max(mileage):.0f} km")
    print(f"   Prix: {np.min(actual_prices):.0f} - {np.max(actual_prices):.0f} €")
    
    # Calculate predictions
    predictions = []
    for km in mileage:
        km_normalized = (km - MEAN_KM) / STD_KM
        price_normalized = THETA0 + (THETA1 * km_normalized)
        pred_price = price_normalized * STD_PRICE + MEAN_PRICE
        predictions.append(pred_price)
    predictions = np.array(predictions)
    
    return mileage, actual_prices, predictions


def plot_data_distribution(mileage, actual_prices):
    """
    Create visualization for data distribution.
    
    Args:
        mileage: Array of mileage values
        actual_prices: Array of actual price values
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("\n 1. VISUALISATION DE LA RÉPARTITION DES DONNÉES")
    print("-" * 50)
    
    # Create comprehensive data visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Main scatter plot showing data distribution
    ax1 = plt.subplot(2, 3, 1)
    scatter = plt.scatter(mileage, actual_prices, c=actual_prices, cmap='viridis', 
                         alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Prix (€)')
    plt.xlabel('Kilométrage (km)', fontsize=11)
    plt.ylabel('Prix (€)', fontsize=11)
    plt.title(' Répartition des Données\nPrix vs Kilométrage', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add data statistics
    stats_text = f'N = {len(mileage)} points\n'
    stats_text += f'Corrélation = {np.corrcoef(mileage, actual_prices)[0,1]:.3f}\n'
    stats_text += f'Étendue km: {np.ptp(mileage):.0f}\n'
    stats_text += f'Étendue prix: {np.ptp(actual_prices):.0f} €'
    plt.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Mileage distribution
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(mileage, bins=12, color='skyblue', alpha=0.7, edgecolor='navy')
    plt.axvline(np.mean(mileage), color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {np.mean(mileage):.0f}')
    plt.xlabel('Kilométrage (km)', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title(' Distribution du Kilométrage', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Price distribution
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(actual_prices, bins=12, color='lightcoral', alpha=0.7, edgecolor='darkred')
    plt.axvline(np.mean(actual_prices), color='blue', linestyle='--', linewidth=2, 
                label=f'Moyenne: {np.mean(actual_prices):.0f}€')
    plt.xlabel('Prix (€)', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title(' Distribution des Prix', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("   ✅ Graphique de répartition des données créé")
    return fig


def add_regression_plot(fig, mileage, actual_prices, predictions):
    """
    Add regression line visualization to existing figure.
    
    Args:
        fig: Matplotlib figure object
        mileage: Array of mileage values
        actual_prices: Array of actual price values
        predictions: Array of predicted price values
    """
    print("\n🎯 2. VISUALISATION DE LA LIGNE DE RÉGRESSION LINÉAIRE")
    print("-" * 55)
    
    # Main regression plot
    ax4 = plt.subplot(2, 3, (4, 6))  # Span bottom row
    
    # Scatter plot of actual data
    plt.scatter(mileage, actual_prices, color='blue', alpha=0.7, s=70, 
               label=' Données réelles', edgecolors='darkblue', linewidth=0.5)
    
    # Create smooth regression line
    km_range = np.linspace(np.min(mileage) - 5000, np.max(mileage) + 5000, 200)
    smooth_predictions = []
    for km in km_range:
        km_normalized = (km - MEAN_KM) / STD_KM
        price_normalized = THETA0 + (THETA1 * km_normalized)
        pred_price = price_normalized * STD_PRICE + MEAN_PRICE
        smooth_predictions.append(pred_price)
    smooth_predictions = np.array(smooth_predictions)
    
    # Plot regression line
    plt.plot(km_range, smooth_predictions, color='red', linewidth=3, 
             label=' Ligne de régression', alpha=0.9)
    
    # Plot individual predictions
    plt.scatter(mileage, predictions, color='orange', alpha=0.8, s=50,
               label=' Prédictions', marker='x', linewidth=2)
    
    # Connect actual to predicted with lines
    for i in range(len(mileage)):
        plt.plot([mileage[i], mileage[i]], [actual_prices[i], predictions[i]], 
                'gray', alpha=0.3, linewidth=1)
    
    plt.xlabel('Kilométrage (km)', fontsize=12)
    plt.ylabel('Prix (€)', fontsize=12)
    plt.title(' Résultat de la Régression Linéaire\n' + 
              f'Équation: prix = f(km_normalisé) où f(x) = {THETA0:.6f} + {THETA1:.6f}×x',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add model information
    model_info = f'Modèle entraîné:\n'
    model_info += f'θ₀ = {THETA0:.6f}\n'
    model_info += f'θ₁ = {THETA1:.6f}\n\n'
    model_info += f'Normalisation:\n'
    model_info += f'μ_km = {MEAN_KM:.0f} km\n'
    model_info += f'σ_km = {STD_KM:.0f} km\n'
    model_info += f'μ_prix = {MEAN_PRICE:.0f} €\n'
    model_info += f'σ_prix = {STD_PRICE:.0f} €'
    
    plt.text(0.02, 0.98, model_info, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('../graphs/complete_demonstration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Graphique de régression linéaire créé")


def calculate_precision_metrics(actual_prices, predictions, mileage):
    """
    Calculate comprehensive precision metrics.
    
    Args:
        actual_prices: Array of actual price values
        predictions: Array of predicted price values
        mileage: Array of mileage values
        
    Returns:
        dict: Dictionary containing all precision metrics
    """
    # Calculate comprehensive precision metrics
    errors = actual_prices - predictions
    abs_errors = np.abs(errors)
    
    # Basic metrics
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(abs_errors / actual_prices) * 100
    
    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Correlation
    correlation = np.corrcoef(mileage, actual_prices)[0, 1]
    
    # Precision thresholds
    within_500 = np.sum(abs_errors <= 500) / len(errors) * 100
    within_1000 = np.sum(abs_errors <= 1000) / len(errors) * 100
    within_1500 = np.sum(abs_errors <= 1500) / len(errors) * 100
    
    # Best and worst predictions
    best_idx = np.argmin(abs_errors)
    worst_idx = np.argmax(abs_errors)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r_squared': r_squared,
        'correlation': correlation,
        'within_500': within_500,
        'within_1000': within_1000,
        'within_1500': within_1500,
        'best_idx': best_idx,
        'worst_idx': worst_idx,
        'errors': errors,
        'abs_errors': abs_errors
    }


def print_precision_results(metrics, mileage, actual_prices, predictions):
    """
    Print detailed precision analysis results.
    
    Args:
        metrics: Dictionary of calculated metrics
        mileage: Array of mileage values
        actual_prices: Array of actual price values
        predictions: Array of predicted price values
    """
    print("\n🔬 3. CALCUL DE LA PRÉCISION DE L'ALGORITHME")
    print("-" * 45)
    
    print(f"\n📊 RÉSULTATS DE PRÉCISION:")
    print(f"   • Mean Absolute Error (MAE):     {metrics['mae']:.2f} €")
    print(f"   • Root Mean Squared Error:       {metrics['rmse']:.2f} €")
    print(f"   • Mean Absolute Percentage:      {metrics['mape']:.2f} %")
    print(f"   • Coefficient R²:                {metrics['r_squared']:.4f} ({metrics['r_squared']*100:.1f}%)")
    print(f"   • Corrélation:                   {metrics['correlation']:.4f}")
    
    print(f"\n PRÉCISION PAR SEUILS:")
    print(f"   • Prédictions à ±500€:           {metrics['within_500']:.1f}% ({int(metrics['within_500']*len(metrics['errors'])/100)}/{len(metrics['errors'])})")
    print(f"   • Prédictions à ±1000€:          {metrics['within_1000']:.1f}% ({int(metrics['within_1000']*len(metrics['errors'])/100)}/{len(metrics['errors'])})")
    print(f"   • Prédictions à ±1500€:          {metrics['within_1500']:.1f}% ({int(metrics['within_1500']*len(metrics['errors'])/100)}/{len(metrics['errors'])})")
    
    # Quality assessment
    print(f"\n📋 ÉVALUATION DE LA QUALITÉ:")
    if metrics['r_squared'] >= 0.8:
        quality = " EXCELLENTE"
    elif metrics['r_squared'] >= 0.6:
        quality = " BONNE"
    elif metrics['r_squared'] >= 0.4:
        quality = " MODÉRÉE"
    else:
        quality = " FAIBLE"
    
    print(f"   • Qualité globale du modèle:     {quality} (R² = {metrics['r_squared']:.3f})")
    
    if metrics['mape'] <= 10:
        error_assessment = " EXCELLENTE"
    elif metrics['mape'] <= 20:
        error_assessment = " ACCEPTABLE"
    else:
        error_assessment = " ÉLEVÉE"
    
    print(f"   • Erreur pourcentage moyenne:    {error_assessment} ({metrics['mape']:.1f}%)")
    
    # Best and worst predictions
    print(f"\n MEILLEURE PRÉDICTION:")
    print(f"   Point #{metrics['best_idx']+1}: {mileage[metrics['best_idx']]:.0f}km → {actual_prices[metrics['best_idx']]:.0f}€ (prédit: {predictions[metrics['best_idx']]:.0f}€, erreur: {metrics['abs_errors'][metrics['best_idx']]:.0f}€)")
    
    print(f"\n  PIRE PRÉDICTION:")
    print(f"   Point #{metrics['worst_idx']+1}: {mileage[metrics['worst_idx']]:.0f}km → {actual_prices[metrics['worst_idx']]:.0f}€ (prédit: {predictions[metrics['worst_idx']]:.0f}€, erreur: {metrics['abs_errors'][metrics['worst_idx']]:.0f}€)")


def create_precision_plots(metrics, actual_prices, predictions):
    """
    Create precision analysis plots.
    
    Args:
        metrics: Dictionary of calculated metrics
        actual_prices: Array of actual price values
        predictions: Array of predicted price values
    """
    # Create precision summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted
    ax1.scatter(actual_prices, predictions, alpha=0.7, s=80, color='blue', edgecolors='darkblue')
    min_price = min(np.min(actual_prices), np.min(predictions))
    max_price = max(np.max(actual_prices), np.max(predictions))
    ax1.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Prédiction parfaite')
    ax1.set_xlabel('Prix Réel (€)')
    ax1.set_ylabel('Prix Prédit (€)')
    ax1.set_title(f' Prédictions vs Réalité\nR² = {metrics["r_squared"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.hist(metrics['errors'], bins=12, alpha=0.7, color='orange', edgecolor='darkorange')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax2.axvline(np.mean(metrics['errors']), color='blue', linestyle='--', linewidth=2, 
                label=f'Erreur moyenne: {np.mean(metrics["errors"]):.0f}€')
    ax2.set_xlabel('Erreur (€)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title(f' Distribution des Erreurs\nMAE = {metrics["mae"]:.0f}€')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Absolute errors by data point
    ax3.bar(range(len(metrics['abs_errors'])), metrics['abs_errors'], alpha=0.7, color='purple')
    ax3.axhline(metrics['mae'], color='red', linestyle='--', linewidth=2, label=f'MAE = {metrics["mae"]:.0f}€')
    ax3.set_xlabel('Index des Points')
    ax3.set_ylabel('Erreur Absolue (€)')
    ax3.set_title(' Erreurs par Point de Données')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Precision metrics summary
    ax4.axis('off')
    
    # Quality assessment for summary
    if metrics['r_squared'] >= 0.8:
        quality = "EXCELLENTE"
    elif metrics['r_squared'] >= 0.6:
        quality = "BONNE"
    elif metrics['r_squared'] >= 0.4:
        quality = "MODÉRÉE"
    else:
        quality = "FAIBLE"
    
    metrics_text = f"""
 RÉSUMÉ DES MÉTRIQUES

 Erreurs:
   MAE:  {metrics['mae']:.0f} €
   RMSE: {metrics['rmse']:.0f} €
   MAPE: {metrics['mape']:.1f} %

 Qualité:
   R²:           {metrics['r_squared']:.4f}
   Corrélation:  {metrics['correlation']:.4f}

 Précision:
   ±500€:  {metrics['within_500']:.0f}%
   ±1000€: {metrics['within_1000']:.0f}%
   ±1500€: {metrics['within_1500']:.0f}%

 Évaluation: {quality}
"""
    ax4.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    #plt.show()
    plt.tight_layout()
    plt.savefig('../graphs/precision_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Analyse de précision terminée")
    print(f"📁 Graphique sauvegardé: ../graphs/precision_summary.png")


def demonstrate_complete_pipeline():
    """
    Complete demonstration of the linear regression project (Refactored).
    """
    print("🚀 DÉMONSTRATION COMPLÈTE DU PROJET ft_linear_regression")
    print("=" * 65)
    
    # 1. Load and prepare data
    data_result = load_and_prepare_data()
    if data_result is None:
        return
    
    mileage, actual_prices, predictions = data_result
    
    # 2. Plot data distribution
    fig = plot_data_distribution(mileage, actual_prices)
    
    # 3. Add regression line to the plot
    add_regression_plot(fig, mileage, actual_prices, predictions)
    
    # 4. Calculate precision metrics
    metrics = calculate_precision_metrics(actual_prices, predictions, mileage)
    
    # 5. Print precision results
    print_precision_results(metrics, mileage, actual_prices, predictions)
    
    # 6. Create precision plots
    create_precision_plots(metrics, actual_prices, predictions)
    
    # Final summary
    quality = "🟢 EXCELLENTE" if metrics['r_squared'] >= 0.8 else \
             "🟡 BONNE" if metrics['r_squared'] >= 0.6 else \
             "🟠 MODÉRÉE" if metrics['r_squared'] >= 0.4 else "🔴 FAIBLE"
    
    print(f"\n🎉 DÉMONSTRATION COMPLÈTE TERMINÉE")
    print("=" * 65)
    print(f"✅ 1. Répartition des données visualisée")
    print(f"✅ 2. Ligne de régression linéaire tracée")
    print(f"✅ 3. Précision de l'algorithme calculée")
    print(f"\n📁 Graphiques générés:")
    print(f"   • ../graphs/complete_demonstration.png")
    print(f"   • ../graphs/precision_summary.png")
    print(f"\n🏆 Qualité du modèle: {quality} (R² = {metrics['r_squared']:.3f})")
    print(f"🎯 Erreur moyenne: {metrics['mae']:.0f}€ ({metrics['mape']:.1f}%)")


if __name__ == "__main__":
    demonstrate_complete_pipeline()
