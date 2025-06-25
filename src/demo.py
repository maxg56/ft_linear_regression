#!/usr/bin/env python3
"""
Complete demonstration script for ft_linear_regression project.
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

def demonstrate_complete_pipeline():
    """
    Complete demonstration of the linear regression project.
    """
    print("🚀 DÉMONSTRATION COMPLÈTE DU PROJET ft_linear_regression")
    print("=" * 65)
    
    # Load data
    data = load_csv("../data/data.csv")
    if data is None or len(data) == 0:
        print("❌ Erreur: Impossible de charger les données")
        return
    
    mileage = data['km'].values
    actual_prices = data['price'].values
    
    print(f"📊 Données chargées: {len(mileage)} points")
    print(f"   Kilométrage: {np.min(mileage):.0f} - {np.max(mileage):.0f} km")
    print(f"   Prix: {np.min(actual_prices):.0f} - {np.max(actual_prices):.0f} €")
    
    # ========================================================================
    # 1. PLOTTING THE DATA DISTRIBUTION
    # ========================================================================
    print("\n📈 1. VISUALISATION DE LA RÉPARTITION DES DONNÉES")
    print("-" * 50)
    
    # Calculate predictions for plotting
    predictions = []
    for km in mileage:
        km_normalized = (km - MEAN_KM) / STD_KM
        price_normalized = THETA0 + (THETA1 * km_normalized)
        pred_price = price_normalized * STD_PRICE + MEAN_PRICE
        predictions.append(pred_price)
    predictions = np.array(predictions)
    
    # Create comprehensive data visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Main scatter plot showing data distribution
    ax1 = plt.subplot(2, 3, 1)
    scatter = plt.scatter(mileage, actual_prices, c=actual_prices, cmap='viridis', 
                         alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Prix (€)')
    plt.xlabel('Kilométrage (km)', fontsize=11)
    plt.ylabel('Prix (€)', fontsize=11)
    plt.title('🔍 Répartition des Données\nPrix vs Kilométrage', fontsize=12, fontweight='bold')
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
    n, bins, patches = plt.hist(mileage, bins=12, color='skyblue', alpha=0.7, edgecolor='navy')
    plt.axvline(np.mean(mileage), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(mileage):.0f}')
    plt.xlabel('Kilométrage (km)', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title('📊 Distribution du Kilométrage', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Price distribution
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(actual_prices, bins=12, color='lightcoral', alpha=0.7, edgecolor='darkred')
    plt.axvline(np.mean(actual_prices), color='blue', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(actual_prices):.0f}€')
    plt.xlabel('Prix (€)', fontsize=11)
    plt.ylabel('Fréquence', fontsize=11)
    plt.title('💰 Distribution des Prix', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("   ✅ Graphique de répartition des données créé")
    
    # ========================================================================
    # 2. PLOTTING THE LINEAR REGRESSION LINE
    # ========================================================================
    print("\n🎯 2. VISUALISATION DE LA LIGNE DE RÉGRESSION LINÉAIRE")
    print("-" * 55)
    
    # Main regression plot
    ax4 = plt.subplot(2, 3, (4, 6))  # Span bottom row
    
    # Scatter plot of actual data
    plt.scatter(mileage, actual_prices, color='blue', alpha=0.7, s=70, 
               label='🔵 Données réelles', edgecolors='darkblue', linewidth=0.5)
    
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
             label='🔴 Ligne de régression', alpha=0.9)
    
    # Plot individual predictions
    plt.scatter(mileage, predictions, color='orange', alpha=0.8, s=50,
               label='🟠 Prédictions', marker='x', linewidth=2)
    
    # Connect actual to predicted with lines
    for i in range(len(mileage)):
        plt.plot([mileage[i], mileage[i]], [actual_prices[i], predictions[i]], 
                'gray', alpha=0.3, linewidth=1)
    
    plt.xlabel('Kilométrage (km)', fontsize=12)
    plt.ylabel('Prix (€)', fontsize=12)
    plt.title('🎯 Résultat de la Régression Linéaire\n' + 
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
    plt.savefig('../graphs/complete_demonstration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Graphique de régression linéaire créé")
    print("   📁 Sauvegardé: ../graphs/complete_demonstration.png")
    
    # ========================================================================
    # 3. CALCULATING ALGORITHM PRECISION
    # ========================================================================
    print("\n🔬 3. CALCUL DE LA PRÉCISION DE L'ALGORITHME")
    print("-" * 45)
    
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
    
    print(f"\n📊 RÉSULTATS DE PRÉCISION:")
    print(f"   • Mean Absolute Error (MAE):     {mae:.2f} €")
    print(f"   • Root Mean Squared Error:       {rmse:.2f} €")
    print(f"   • Mean Absolute Percentage:      {mape:.2f} %")
    print(f"   • Coefficient R²:                {r_squared:.4f} ({r_squared*100:.1f}%)")
    print(f"   • Corrélation:                   {correlation:.4f}")
    
    print(f"\n🎯 PRÉCISION PAR SEUILS:")
    print(f"   • Prédictions à ±500€:           {within_500:.1f}% ({int(within_500*len(errors)/100)}/{len(errors)})")
    print(f"   • Prédictions à ±1000€:          {within_1000:.1f}% ({int(within_1000*len(errors)/100)}/{len(errors)})")
    print(f"   • Prédictions à ±1500€:          {within_1500:.1f}% ({int(within_1500*len(errors)/100)}/{len(errors)})")
    
    # Quality assessment
    print(f"\n📋 ÉVALUATION DE LA QUALITÉ:")
    if r_squared >= 0.8:
        quality = "🟢 EXCELLENTE"
    elif r_squared >= 0.6:
        quality = "🟡 BONNE"
    elif r_squared >= 0.4:
        quality = "🟠 MODÉRÉE"
    else:
        quality = "🔴 FAIBLE"
    
    print(f"   • Qualité globale du modèle:     {quality} (R² = {r_squared:.3f})")
    
    if mape <= 10:
        error_assessment = "🟢 EXCELLENTE"
    elif mape <= 20:
        error_assessment = "🟡 ACCEPTABLE"
    else:
        error_assessment = "🔴 ÉLEVÉE"
    
    print(f"   • Erreur pourcentage moyenne:    {error_assessment} ({mape:.1f}%)")
    
    # Best and worst predictions
    best_idx = np.argmin(abs_errors)
    worst_idx = np.argmax(abs_errors)
    
    print(f"\n🏆 MEILLEURE PRÉDICTION:")
    print(f"   Point #{best_idx+1}: {mileage[best_idx]:.0f}km → {actual_prices[best_idx]:.0f}€ (prédit: {predictions[best_idx]:.0f}€, erreur: {abs_errors[best_idx]:.0f}€)")
    
    print(f"\n⚠️  PIRE PRÉDICTION:")
    print(f"   Point #{worst_idx+1}: {mileage[worst_idx]:.0f}km → {actual_prices[worst_idx]:.0f}€ (prédit: {predictions[worst_idx]:.0f}€, erreur: {abs_errors[worst_idx]:.0f}€)")
    
    # Create precision summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted
    ax1.scatter(actual_prices, predictions, alpha=0.7, s=80, color='blue', edgecolors='darkblue')
    min_price = min(np.min(actual_prices), np.min(predictions))
    max_price = max(np.max(actual_prices), np.max(predictions))
    ax1.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Prédiction parfaite')
    ax1.set_xlabel('Prix Réel (€)')
    ax1.set_ylabel('Prix Prédit (€)')
    ax1.set_title(f'🎯 Prédictions vs Réalité\nR² = {r_squared:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.hist(errors, bins=12, alpha=0.7, color='orange', edgecolor='darkorange')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax2.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2, label=f'Erreur moyenne: {np.mean(errors):.0f}€')
    ax2.set_xlabel('Erreur (€)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title(f'📊 Distribution des Erreurs\nMAE = {mae:.0f}€')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Absolute errors by data point
    ax3.bar(range(len(abs_errors)), abs_errors, alpha=0.7, color='purple')
    ax3.axhline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.0f}€')
    ax3.set_xlabel('Index des Points')
    ax3.set_ylabel('Erreur Absolue (€)')
    ax3.set_title('📈 Erreurs par Point de Données')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Precision metrics summary
    ax4.axis('off')
    metrics_text = f"""
📋 RÉSUMÉ DES MÉTRIQUES

🎯 Erreurs:
   MAE:  {mae:.0f} €
   RMSE: {rmse:.0f} €
   MAPE: {mape:.1f} %

📊 Qualité:
   R²:           {r_squared:.4f}
   Corrélation:  {correlation:.4f}

✅ Précision:
   ±500€:  {within_500:.0f}%
   ±1000€: {within_1000:.0f}%
   ±1500€: {within_1500:.0f}%

🏆 Évaluation: {quality.split()[-1]}
"""
    ax4.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../graphs/precision_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Analyse de précision terminée")
    print(f"📁 Graphique sauvegardé: ../graphs/precision_summary.png")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n🎉 DÉMONSTRATION COMPLÈTE TERMINÉE")
    print("=" * 65)
    print(f"✅ 1. Répartition des données visualisée")
    print(f"✅ 2. Ligne de régression linéaire tracée")
    print(f"✅ 3. Précision de l'algorithme calculée")
    print(f"\n📁 Graphiques générés:")
    print(f"   • ../graphs/complete_demonstration.png")
    print(f"   • ../graphs/precision_summary.png")
    print(f"\n🏆 Qualité du modèle: {quality} (R² = {r_squared:.3f})")
    print(f"🎯 Erreur moyenne: {mae:.0f}€ ({mape:.1f}%)")

if __name__ == "__main__":
    demonstrate_complete_pipeline()
