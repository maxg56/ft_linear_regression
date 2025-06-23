#!/usr/bin/env python3
"""
Test script for the linear regression model.
This script tests the trained model with various inputs.
"""

from estimatePrice import estimate_price
import numpy as np
from loode_csv import loode_csv

def test_model():
    """
    Test the trained model with various inputs and compare with actual data.
    """
    print("=== Testing Linear Regression Model ===")
    
    # Test with some sample values
    test_mileages = [50000, 100000, 150000, 200000, 250000]
    
    print("\nTesting with sample mileages:")
    for km in test_mileages:
        try:
            price = estimate_price(km)
            print(f"  {km:6d} km -> {price:8.2f} €")
        except Exception as e:
            print(f"  {km:6d} km -> Error: {e}")
    
    # Compare with actual data
    print("\nComparing with actual training data:")
    try:
        data = loode_csv("../data/data.csv")
        if data is not None and len(data) > 0:
            mileages = data['km'].values
            actual_prices = data['price'].values
            
            print(f"{'Mileage':>8} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
            print("-" * 36)
            
            total_error = 0
            for i, (km, actual) in enumerate(zip(mileages, actual_prices)):
                try:
                    predicted = estimate_price(km)
                    error = abs(actual - predicted)
                    total_error += error
                    print(f"{km:8.0f} {actual:8.0f} {predicted:10.2f} {error:8.2f}")
                except Exception as e:
                    print(f"{km:8.0f} {actual:8.0f} {'Error':>10} {str(e):>8}")
            
            avg_error = total_error / len(mileages)
            print("-" * 36)
            print(f"Average absolute error: {avg_error:.2f} €")
            
    except Exception as e:
        print(f"Error loading test data: {e}")
    
    print("\n✅ Testing completed!")

def interactive_test():
    """
    Interactive testing mode where user can input mileage values.
    """
    print("\n=== Interactive Testing Mode ===")
    print("Enter mileage values to get price estimates (type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter mileage (km): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            km = float(user_input)
            price = estimate_price(km)
            print(f"Estimated price for {km:.0f} km: {price:.2f} €")
            
        except ValueError:
            print("Please enter a valid number or 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    test_model()
    
    # Ask if user wants interactive mode
    while True:
        choice = input("\nDo you want to try interactive testing? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_test()
            break
        elif choice in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")
