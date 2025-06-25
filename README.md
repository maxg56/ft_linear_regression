# ft_linear_regression

A linear regression implementation from scratch to predict car prices based on mileage.

## Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv                    # Training data (mileage vs price)
├── graphs/                         # Generated visualization plots
│   ├── complete_demonstration.png  # Complete demo visualization
│   └── precision_summary.png       # Precision analysis summary
├── src/
│   ├── const.py                    # Learned parameters and normalization stats
│   ├── demo.py                     # Complete demonstration script
│   ├── estimatePrice.py            # Price estimation script
│   ├── linear_regression.py        # Core regression implementation
│   └── utils/
│       ├── load_csv.py             # CSV loading utility
│       └── update_constants.py     # Constants update utility
├── Makefile                        # Build automation
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Features

- **Linear regression from scratch** using gradient descent
- **Data normalization** for better convergence
- **Complete data visualization** including:
  - Data distribution and repartition plots
  - Scatter plot with regression line
  - Cost function evolution during training
  - Residual analysis
  - Comparison of original vs normalized data
- **Comprehensive precision calculation** with metrics (MSE, RMSE, MAE, R², MAPE)
- **Interactive testing** mode
- **Command-line price estimation**
- **Complete demonstration** combining all features

## Quick Start

### Complete Demonstration
Run the complete demonstration to see all features:

```bash
make demo
```

This will:
1. **Plot data distribution** - Shows how the data points are spread
2. **Plot regression line** - Shows the result of your linear regression
3. **Calculate precision** - Comprehensive algorithm accuracy analysis

## Usage

### 1. Complete Demonstration

Run the complete demonstration to see all features:

```bash
make demo
```

This will:
- Load data from `data/data.csv`
- Perform linear regression training
- Display data visualization
- Calculate precision metrics
- Generate demonstration plots

### 2. Estimate Car Prices

Use the trained model to estimate prices:

```bash
make estimate KM=<mileage_in_km>
```

Example:
```bash
make estimate KM=50000
# Output: Estimated price for 50000.0 km: 7427.10
```

Or run directly:
```bash
cd src
python estimatePrice.py <mileage_in_km>
```

### 3. Run Core Linear Regression

You can run the core linear regression implementation directly:

```bash
cd src
python linear_regression.py
```

### 4. Clean Generated Files

Remove generated files and cache:

```bash
make clean
```

## Makefile Commands

For easier project management, use the Makefile:

```bash
# Complete demonstration (recommended)
make demo

# Individual components
make train              # Train the linear regression model
make estimate KM=50000  # Estimate price for specific mileage

# Project management
make install            # Install dependencies
make clean              # Clean generated files
make help               # Show all available commands
```

## Project Requirements Fulfilled

This project fulfills all the specified requirements:

### ✅ 1. Plotting the data into a graph to see their repartition
- **File**: `src/demo.py`
- **Command**: `make demo`
- **Output**: Data distribution plots showing:
  - Scatter plot with color-coded prices
  - Data point distribution analysis
  - Statistical information display
- **Graphs**: `complete_demonstration.png`

### ✅ 2. Plotting the line resulting from linear regression
- **File**: `src/demo.py` 
- **Command**: `make demo`
- **Output**: Regression visualization showing:
  - Original data points
  - Linear regression line
  - Model equation and parameters
  - Correlation information
- **Graphs**: `complete_demonstration.png`

### ✅ 3. A program that calculates the precision of your algorithm
- **File**: `src/demo.py`
- **Command**: `make demo`
- **Output**: Comprehensive precision metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - R-squared coefficient
  - Quality assessment
- **Graphs**: `precision_summary.png`

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
make install
```

Or manually:
```bash
pip install -r requirements.txt
```

## Files Description

### Core Files
- **`demo.py`**: Complete demonstration script (all requirements in one)
- **`estimatePrice.py`**: Command-line price estimation tool
- **`linear_regression.py`**: Core gradient descent implementation
- **`const.py`**: Stores learned parameters and normalization statistics

### Utility Files
- **`utils/load_csv.py`**: CSV file loading utility
- **`utils/update_constants.py`**: Constants update utility

### Data Files
- **`data/data.csv`**: Training dataset
- **`graphs/*.png`**: Generated visualization plots:
  - `complete_demonstration.png`: Main demo output
  - `precision_summary.png`: Precision metrics summary

### Configuration Files
- **`Makefile`**: Build automation and project commands
- **`requirements.txt`**: Python dependencies
- **`README.md`**: Project documentation

## Example Output

### Complete Demonstration
```bash
$ make demo
🚀 DÉMONSTRATION COMPLÈTE DU PROJET ft_linear_regression
📊 Données chargées: 24 points
✅ 1. Répartition des données visualisée
✅ 2. Ligne de régression linéaire tracée  
✅ 3. Précision de l'algorithme calculée
🏆 Qualité du modèle: 🟡 BONNE (R² = 0.733)
🎯 Erreur moyenne: 558€ (9.6%)
```

### Price Estimation
```bash
$ make estimate KM=100000
Estimating price for 100000 km...
Estimated price for 100000.0 km: 6354.70
```

### Training Output
```bash
$ make train
Training linear regression model...
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
```

## Contributing

This is an educational project implementing linear regression from scratch. Feel free to experiment with different:
- Learning rates
- Number of iterations
- Feature engineering approaches
- Visualization styles

## License

Educational project - feel free to use and modify.
