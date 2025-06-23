import numpy as np
from loode_csv import loode_csv

def update_constants(theta: np.ndarray, mean_km: float, std_km: float, mean_price: float, std_price: float):
    """
    Update the constants in const.py with the learned parameters and normalization statistics.
    
    :param theta: Learned parameters from gradient descent
    :param mean_km: Mean of mileage data used for normalization
    :param std_km: Standard deviation of mileage data used for normalization
    :param mean_price: Mean of price data used for normalization
    :param std_price: Standard deviation of price data used for normalization
    """
    const_file_path = "const.py"
    
    content = f"""# Learned parameters from linear regression (normalized data)
THETA0 = {theta[0]}  # Intercept (theta[0])
THETA1 = {theta[1]}  # Slope (theta[1])

# Normalization statistics from training data
MEAN_KM = {mean_km}      # Mean of mileage data
STD_KM = {std_km}        # Standard deviation of mileage data  
MEAN_PRICE = {mean_price}   # Mean of price data
STD_PRICE = {std_price}     # Standard deviation of price data
"""
    
    # Write the updated content to the file
    with open(const_file_path, 'w') as file:
        file.write(content)
    
    print(f"Constants updated in {const_file_path}")
    print(f"THETA0 = {theta[0]}")
    print(f"THETA1 = {theta[1]}")
    print(f"MEAN_KM = {mean_km}")
    print(f"STD_KM = {std_km}")
    print(f"MEAN_PRICE = {mean_price}")
    print(f"STD_PRICE = {std_price}")


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
    data = loode_csv("../data/data.csv")
    
    # Extract features and target from loaded data
    # Assuming data has columns for mileage and price
    if data is not None and len(data) > 0:
        # Extract mileage (feature) and add bias term
        mileage = data['km'].values  # Extract km column as numpy array
        price = data['price'].values  # Extract price column as numpy array
        
        # Store original statistics for denormalization
        mean_km = np.mean(mileage)
        std_km = np.std(mileage)
        mean_price = np.mean(price)
        std_price = np.std(price)
        
        # Normalize the features to prevent overflow
        mileage_normalized = (mileage - mean_km) / std_km
        price_normalized = (price - mean_price) / std_price
        
        X = np.column_stack([np.ones(len(mileage_normalized)), mileage_normalized])  # Add bias term
        y = price_normalized
    else:
        print("Warning: Using example data as CSV couldn't be loaded")
        return
    
    
    # Initial parameters (theta)
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
    
    # Update constants in const.py with the learned parameters and normalization stats
    update_constants(theta, mean_km, std_km, mean_price, std_price)

if __name__ == "__main__":
    main()
