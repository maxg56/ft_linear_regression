#!/usr/bin/env python3
"""
Training script for the linear regression model.
This script trains the model and saves the learned parameters to const.py
"""

import numpy as np
from loode_csv import loode_csv
from linear_regression import Gradient_descent, update_constants

def train_model():
    """
    Train the linear regression model and save parameters to const.py
    """
    print("=== Training Linear Regression Model ===")
    
    # Load data from CSV file
    data = loode_csv("../data/data.csv")
    
    if data is None or len(data) == 0:
        print("Error: Could not load data from CSV file")
        return False
    
    print(f"Loaded {len(data)} data points")
    
    # Extract features and target
    mileage = data['km'].values
    price = data['price'].values
    
    # Store original statistics for denormalization
    mean_km = np.mean(mileage)
    std_km = np.std(mileage)
    mean_price = np.mean(price)
    std_price = np.std(price)
    
    print(f"Data statistics:")
    print(f"  Mileage: mean={mean_km:.2f}, std={std_km:.2f}")
    print(f"  Price: mean={mean_price:.2f}, std={std_price:.2f}")
    
    # Normalize the features for training
    mileage_normalized = (mileage - mean_km) / std_km
    price_normalized = (price - mean_price) / std_price
    
    # Create feature matrix with bias term
    X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])
    y = price_normalized
    
    # Initialize parameters
    theta = np.zeros(X.shape[1])
    alpha = 0.01
    iterations = 1000
    
    print(f"\nTraining parameters:")
    print(f"  Learning rate: {alpha}")
    print(f"  Iterations: {iterations}")
    
    # Train the model
    print("\nTraining model...")
    theta_trained = Gradient_descent(X, y, theta, alpha, iterations)
    
    # Calculate training accuracy
    predictions_normalized = X.dot(theta_trained)
    predictions = predictions_normalized * std_price + mean_price
    
    # Calculate metrics
    mse = np.mean((price - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(price - predictions))
    
    # R-squared
    ss_res = np.sum((price - predictions) ** 2)
    ss_tot = np.sum((price - np.mean(price)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nTraining Results:")
    print(f"  Final parameters: θ₀={theta_trained[0]:.6f}, θ₁={theta_trained[1]:.6f}")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  Root Mean Squared Error: {rmse:.2f}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  R-squared: {r_squared:.4f}")
    
    # Save parameters to const.py
    print("\nSaving parameters to const.py...")
    update_constants(theta_trained, mean_km, std_km, mean_price, std_price)
    
    print("\n✅ Training completed successfully!")
    return True

if __name__ == "__main__":
    train_model()
