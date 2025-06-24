from const import THETA0, THETA1, MEAN_KM, STD_KM, MEAN_PRICE, STD_PRICE
import sys

def estimate_price(km: float) -> float:
    """
    Estimate the price of a car based on the mileage in kilometers.
    
    :param km: Mileage in kilometers
    :return: Estimated price
    """
    if km < 0:
        raise ValueError("Kilometers cannot be negative.")
    
    km_normalized = (km - MEAN_KM) / STD_KM

    price_normalized = THETA0 + (THETA1 * km_normalized)
    
    return  price_normalized * STD_PRICE + MEAN_PRICE



def main():
    if len(sys.argv) < 2:
        print("Usage: python estimatePrice.py <kilometers>")
        return
    
    km = sys.argv[1]
    try:
        if not km.replace('.', '', 1).isdigit():
            raise ValueError("Invalid input: Please enter a numeric value for kilometers.")
        km = float(km)
        price = estimate_price(km)
        print(f"Estimated price for {km} km: {price:.2f}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

