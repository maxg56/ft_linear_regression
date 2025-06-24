import numpy as np

CONST_FILE_PATH = "const.py"

def update_constants(theta: np.ndarray, mean_km: float, std_km: float, mean_price: float, std_price: float):
    """
    Update the constants in const.py with the learned parameters and normalization statistics.
    
    :param theta: Learned parameters from gradient descent
    :param mean_km: Mean of mileage data used for normalization
    :param std_km: Standard deviation of mileage data used for normalization
    :param mean_price: Mean of price data used for normalization
    :param std_price: Standard deviation of price data used for normalization
    """
    with open(CONST_FILE_PATH, "w") as f:
        f.write(f'THETA0 = "{theta[0]}"\n')
        f.write(f'THETA1 = "{theta[1]}"\n')
        f.write(f'MEAN_KM = "{mean_km}"\n')
        f.write(f'STD_KM = "{std_km}"\n')
        f.write(f'MEAN_PRICE = "{mean_price}"\n')
        f.write(f'STD_PRICE = "{std_price}"\n')