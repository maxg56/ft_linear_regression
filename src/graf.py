import matplotlib.pyplot as plt
import numpy as np
from loode_csv import loode_csv
from linear_regression import Gradient_descent, estimate_price

def plot_regression():
    """
    Visualize the linear regression results with scatter plot and regression line.
    """
    # Load data from CSV file
    data = loode_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return
    
    # Extract features and target
    mileage = data['km'].values
    price = data['price'].values
    
    # Normalize the features for training
    mileage_normalized = (mileage - np.mean(mileage)) / np.std(mileage)
    price_normalized = (price - np.mean(price)) / np.std(price)
    
    # Create feature matrix with bias term
    X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])
    y = price_normalized
    
    # Train the model
    theta = np.zeros(X.shape[1])
    alpha = 0.01
    iterations = 1000
    
    theta_trained = Gradient_descent(X, y, theta, alpha, iterations)
    
    # Make predictions
    predictions_normalized = estimate_price(X, theta_trained)
    
    # Denormalize predictions
    predictions = predictions_normalized * np.std(price) + np.mean(price)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of actual data
    plt.scatter(mileage, price, color='blue', alpha=0.6, label='Données réelles', s=50)
    
    # Sort data for smooth line plotting
    sorted_indices = np.argsort(mileage)
    mileage_sorted = mileage[sorted_indices]
    predictions_sorted = predictions[sorted_indices]
    
    # Plot regression line
    plt.plot(mileage_sorted, predictions_sorted, color='red', linewidth=2, label='Ligne de régression')
    
    # Add labels and title
    plt.xlabel('Kilométrage (km)', fontsize=12)
    plt.ylabel('Prix (€)', fontsize=12)
    plt.title('Régression Linéaire - Prix des Voitures vs Kilométrage', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with model parameters
    textstr = f'θ₀ (intercept): {theta_trained[0]:.6f}\nθ₁ (slope): {theta_trained[1]:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Calculate and display R-squared
    ss_res = np.sum((price - predictions) ** 2)
    ss_tot = np.sum((price - np.mean(price)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    plt.text(0.05, 0.80, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
             fontsize=10, bbox=props)
    
    plt.tight_layout()
    plt.savefig('../graphs/regression_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique sauvegardé: ../graphs/regression_plot.png")

def plot_cost_function():
    """
    Visualize the cost function evolution during gradient descent.
    """
    # Load data
    data = loode_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return
    
    # Prepare data
    mileage = data['km'].values
    price = data['price'].values
    
    mileage_normalized = (mileage - np.mean(mileage)) / np.std(mileage)
    price_normalized = (price - np.mean(price)) / np.std(price)
    
    X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])
    y = price_normalized
    
    # Modified gradient descent to track cost
    theta = np.zeros(X.shape[1])
    alpha = 0.01
    iterations = 1000
    costs = []
    
    m = len(y)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        cost = (1/(2*m)) * np.sum(errors**2)
        costs.append(cost)
        
        gradient = (1/m) * X.T.dot(errors)
        theta -= alpha * gradient
    
    # Plot cost function
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), costs, color='blue', linewidth=2)
    plt.xlabel('Itérations', fontsize=12)
    plt.ylabel('Fonction de Coût', fontsize=12)
    plt.title('Évolution de la Fonction de Coût pendant l\'Entraînement', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../graphs/cost_function.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique sauvegardé: ../graphs/cost_function.png")

def plot_residuals():
    """
    Plot residuals to analyze model performance.
    """
    # Load and prepare data
    data = loode_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return
    
    mileage = data['km'].values
    price = data['price'].values
    
    mileage_normalized = (mileage - np.mean(mileage)) / np.std(mileage)
    price_normalized = (price - np.mean(price)) / np.std(price)
    
    X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])
    y = price_normalized
    
    # Train model
    theta = np.zeros(X.shape[1])
    theta_trained = Gradient_descent(X, y, theta, 0.01, 1000)
    
    # Calculate residuals
    predictions_normalized = estimate_price(X, theta_trained)
    predictions = predictions_normalized * np.std(price) + np.mean(price)
    residuals = price - predictions
    
    # Plot residuals
    plt.figure(figsize=(12, 5))
    
    # Residuals vs Fitted values
    plt.subplot(1, 2, 1)
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Résidus')
    plt.title('Résidus vs Valeurs Prédites')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Résidus')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../graphs/residuals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique sauvegardé: ../graphs/residuals_analysis.png")

def plot_comparison():
    """
    Compare original data with normalized data and predictions.
    """
    data = loode_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return
    
    mileage = data['km'].values
    price = data['price'].values
    
    # Normalize
    mileage_normalized = (mileage - np.mean(mileage)) / np.std(mileage)
    price_normalized = (price - np.mean(price)) / np.std(price)
    
    # Train model
    X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])
    y = price_normalized
    theta = np.zeros(X.shape[1])
    theta_trained = Gradient_descent(X, y, theta, 0.01, 1000)
    
    # Predictions
    predictions_normalized = estimate_price(X, theta_trained)
    predictions = predictions_normalized * np.std(price) + np.mean(price)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data
    axes[0].scatter(mileage, price, color='blue', alpha=0.6, label='Données réelles')
    sorted_indices = np.argsort(mileage)
    axes[0].plot(mileage[sorted_indices], predictions[sorted_indices], 
                color='red', linewidth=2, label='Ligne de régression')
    axes[0].set_xlabel('Kilométrage (km)')
    axes[0].set_ylabel('Prix (€)')
    axes[0].set_title('Données Originales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Normalized data
    axes[1].scatter(mileage_normalized, price_normalized, color='green', alpha=0.6, label='Données normalisées')
    sorted_indices_norm = np.argsort(mileage_normalized)
    axes[1].plot(mileage_normalized[sorted_indices_norm], predictions_normalized[sorted_indices_norm], 
                color='red', linewidth=2, label='Ligne de régression')
    axes[1].set_xlabel('Kilométrage (normalisé)')
    axes[1].set_ylabel('Prix (normalisé)')
    axes[1].set_title('Données Normalisées')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../graphs/comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Graphique sauvegardé: ../graphs/comparison_plot.png")

def main():
    """
    Main function to run all visualizations.
    """
    print("=== Visualisation de la Régression Linéaire ===")
    
    try:
        print("\n1. Graphique de régression...")
        plot_regression()
        
        print("\n2. Évolution de la fonction de coût...")
        plot_cost_function()
        
        print("\n3. Analyse des résidus...")
        plot_residuals()
        
        print("\n4. Comparaison données originales/normalisées...")
        plot_comparison()
        
        print("\n✅ Tous les graphiques ont été générés avec succès dans le dossier ../graphs/")
        
    except ImportError:
        print("Erreur: matplotlib n'est pas installé.")
        print("Installez-le avec: pip install matplotlib")
    except Exception as e:
        print(f"Erreur lors de la visualisation: {e}")

if __name__ == "__main__":
    main()