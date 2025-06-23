# ft_linear_regression

A linear regression implementation from scratch to predict car prices based on mileage.

## Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv                 # Training data (mileage vs price)
├── graphs/                      # Generated visualization plots
│   ├── regression_plot.png      # Main regression visualization
│   ├── cost_function.png        # Cost function evolution
│   ├── residuals_analysis.png   # Residual analysis plots
│   └── comparison_plot.png      # Original vs normalized data
└── src/
    ├── const.py                 # Learned parameters and normalization stats
    ├── estimatePrice.py         # Price estimation script
    ├── graf.py                  # Visualization functions
    ├── linear_regression.py     # Core regression implementation
    ├── loode_csv.py            # CSV loading utility
    ├── train.py                # Model training script
    └── test.py                 # Model testing script
```

## Features

- **Linear regression from scratch** using gradient descent
- **Data normalization** for better convergence
- **Comprehensive visualizations** including:
  - Scatter plot with regression line
  - Cost function evolution during training
  - Residual analysis
  - Comparison of original vs normalized data
- **Model evaluation** with metrics (MSE, RMSE, MAE, R²)
- **Interactive testing** mode
- **Command-line price estimation**

## Usage

### 1. Train the Model

Train the linear regression model and save parameters:

```bash
cd src
python train.py
```

This will:
- Load data from `../data/data.csv`
- Train the model using gradient descent
- Save learned parameters to `const.py`
- Display training metrics

### 2. Estimate Car Prices

Use the trained model to estimate prices:

```bash
cd src
python estimatePrice.py <mileage_in_km>
```

Example:
```bash
python estimatePrice.py 50000
# Output: Estimated price for 50000.0 km: 7427.10
```

### 3. Test the Model

Run comprehensive tests on the trained model:

```bash
cd src
python test.py
```

This will:
- Test with sample mileage values
- Compare predictions with actual training data
- Show average prediction error
- Offer interactive testing mode

### 4. Generate Visualizations

Create all visualization plots:

```bash
cd src
python graf.py
```

This generates:
- `regression_plot.png`: Main scatter plot with regression line
- `cost_function.png`: Cost function evolution during training
- `residuals_analysis.png`: Residual analysis plots
- `comparison_plot.png`: Original vs normalized data comparison

### 5. Run Core Functions

You can also run the core linear regression implementation directly:

```bash
cd src
python linear_regression.py
```

## Model Details

### Algorithm
- **Linear Regression** with gradient descent optimization
- **Feature normalization** using z-score standardization
- **Cost function**: Mean Squared Error (MSE)

### Training Parameters
- Learning rate (α): 0.01
- Iterations: 1000
- Features: Mileage (km) with bias term

### Model Equation
After training, the model predicts prices using:

```
price_normalized = θ₀ + θ₁ × mileage_normalized
price = price_normalized × std_price + mean_price
```

Where:
- `θ₀` (theta0): Intercept parameter
- `θ₁` (theta1): Slope parameter
- Normalization statistics are stored in `const.py`

## Data Format

The training data (`data/data.csv`) should have the following format:

```csv
km,price
240000,3650
139800,3800
150500,4400
...
```

- `km`: Mileage in kilometers
- `price`: Car price in euros

## Model Performance

The current model achieves:
- **R² Score**: ~0.73 (73% of variance explained)
- **RMSE**: ~668 euros
- **MAE**: ~558 euros

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib

Install dependencies:
```bash
pip install numpy pandas matplotlib
```

## Files Description

### Core Files
- **`train.py`**: Main training script with comprehensive metrics
- **`estimatePrice.py`**: Command-line price estimation tool
- **`linear_regression.py`**: Core gradient descent implementation
- **`const.py`**: Stores learned parameters and normalization statistics

### Utility Files
- **`loode_csv.py`**: CSV file loading utility
- **`graf.py`**: Comprehensive visualization functions
- **`test.py`**: Model testing and validation

### Data Files
- **`data/data.csv`**: Training dataset (24 data points)
- **`graphs/*.png`**: Generated visualization plots

## Example Output

```bash
$ python train.py
=== Training Linear Regression Model ===
Loaded 24 data points
Data statistics:
  Mileage: mean=101066.25, std=51565.19
  Price: mean=6331.83, std=1291.87

Training Results:
  Final parameters: θ₀=0.000000, θ₁=-0.856102
  Mean Squared Error: 445645.25
  Root Mean Squared Error: 667.57
  Mean Absolute Error: 557.83
  R-squared: 0.7330

✅ Training completed successfully!

$ python estimatePrice.py 100000
Estimated price for 100000.0 km: 6354.70
```

## Contributing

This is an educational project implementing linear regression from scratch. Feel free to experiment with different:
- Learning rates
- Number of iterations
- Feature engineering approaches
- Visualization styles

## License

Educational project - feel free to use and modify.
