import numpy as np
from loode_csv import loode_csv
from const import DATA_FILE
from utils.update_constants import update_constants


def Gradient_descent(X, y, theta, alpha, iterations):
    """
    Perform gradient descent to learn theta.
    
    :param X: Feature matrix
    :param y: Target variable
    :param theta: Initial parameters
    :param alpha: Learning rate
    :param iterations: Number of iterations
    :return: Learned parameters
    """
    m = len(y)
    
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta -= alpha * gradient
    
    return theta

def estimate_price(X, theta):
    """
    Estimate the price based on the feature matrix and learned parameters.
    
    :param X: Feature matrix
    :param theta: Learned parameters
    :return: Estimated prices
    """
    return X.dot(theta)

def main():
    # Load data from CSV file
    data = loode_csv(DATA_FILE)

    if data is not None and len(data) > 0:
        mileage = data['km'].values
        price = data['price'].values
        
        # Store original statistics for denormalization
        mean_km = np.mean(mileage)
        std_km = np.std(mileage)
        mean_price = np.mean(price)
        std_price = np.std(price)
        
        mileage_normalized = (mileage - mean_km) / std_km
        price_normalized = (price - mean_price) / std_price
        
        X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])  # Add bias term
        y = price_normalized
    else:
        print("Warning: Using example data as CSV couldn't be loaded")
        return
    
    
    theta = np.zeros(X.shape[1])
    
    alpha = 0.01
    iterations = 1000
    
    theta = Gradient_descent(X, y, theta, alpha, iterations)
    
    estimated_prices = estimate_price(X, theta)
    
    # Denormalize the results for display
    estimated_prices_denormalized = estimated_prices * std_price + mean_price
    
    print("Learned parameters:", theta)
    print("Estimated prices (normalized):", estimated_prices)
    print("Estimated prices (actual scale):", estimated_prices_denormalized)
    print("Actual prices:", data['price'].values)
    
    update_constants(theta, mean_km, std_km, mean_price, std_price)

if __name__ == "__main__":
    main()
