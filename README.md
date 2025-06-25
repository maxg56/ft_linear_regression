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
cd /path/to/ft_linear_regression
make demo
```

This will:
1. **Plot data distribution** - Shows how the data points are spread
2. **Plot regression line** - Shows the result of your linear regression
3. **Calculate precision** - Comprehensive algorithm accuracy analysis

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

### 4. Plot Data Distribution and Regression Line

Show data repartition and the linear regression result:

```bash
cd src
python plot_data.py
```

This generates:
- `data_distribution.png`: Data distribution analysis with histograms
- `regression_result.png`: Data points with the linear regression line

### 5. Calculate Algorithm Precision

Calculate comprehensive precision metrics:

```bash
cd src
python precision.py
```

This provides:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE) 
- Mean Absolute Percentage Error (MAPE)
- R-squared coefficient
- Precision by error thresholds
- Detailed error analysis

### 6. Complete Demonstration

Run the complete pipeline demonstrating all requirements:

```bash
cd src
python demo.py
```

This combines:
1. **Data repartition plotting**
2. **Linear regression line visualization** 
3. **Algorithm precision calculation**

### 7. Generate All Visualizations

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

### 8. Run Core Functions

You can also run the core linear regression implementation directly:

```bash
cd src
python linear_regression.py
```

## Makefile Commands

For easier project management, use the Makefile:

```bash
# Complete demonstration (recommended)
make demo

# Individual components
make train          # Train the model
make plot           # Plot data distribution and regression line
make precision      # Calculate algorithm precision
make visualize      # Generate all visualization plots
make test           # Test the model
make estimate KM=50000  # Estimate price for specific mileage

# Project management
make install        # Install dependencies
make clean          # Clean generated files
make all            # Train + plot + precision + visualize
make help           # Show all available commands
```

## Project Requirements Fulfilled

This project fulfills all the specified requirements:

### ✅ 1. Plotting the data into a graph to see their repartition
- **File**: `src/plot_data.py` and `src/demo.py`
- **Command**: `make plot` or `make demo`
- **Output**: Data distribution plots showing:
  - Scatter plot with color-coded prices
  - Mileage histogram with mean indicator
  - Price histogram with mean indicator  
  - Box plots for outlier detection
- **Graphs**: `data_distribution.png`, `complete_demonstration.png`

### ✅ 2. Plotting the line resulting from linear regression
- **File**: `src/plot_data.py` and `src/demo.py` 
- **Command**: `make plot` or `make demo`
- **Output**: Regression visualization showing:
  - Original data points (blue)
  - Linear regression line (red)
  - Individual predictions (orange)
  - Model equation and parameters
  - Correlation coefficient
- **Graphs**: `regression_result.png`, `complete_demonstration.png`

### ✅ 3. A program that calculates the precision of your algorithm
- **File**: `src/precision.py` and `src/demo.py`
- **Command**: `make precision` or `make demo`
- **Output**: Comprehensive precision metrics:
  - Mean Absolute Error (MAE): 558€
  - Root Mean Squared Error (RMSE): 668€
  - Mean Absolute Percentage Error (MAPE): 9.6%
  - R-squared coefficient: 0.733 (73.3%)
  - Precision by thresholds (±500€, ±1000€, ±1500€)
  - Best/worst predictions analysis
  - Quality assessment
- **Graphs**: `precision_analysis.png`, `precision_summary.png`

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
- **`demo.py`**: Complete demonstration script (all requirements in one)
- **`plot_data.py`**: Data distribution and regression line visualization
- **`precision.py`**: Algorithm precision calculation and analysis
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
- **`graphs/*.png`**: Generated visualization plots:
  - `complete_demonstration.png`: Main demo output
  - `data_distribution.png`: Data repartition analysis
  - `regression_result.png`: Regression line visualization
  - `precision_summary.png`: Precision metrics summary
  - `precision_analysis.png`: Detailed precision analysis
  - Plus additional visualization plots

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

### Training Output

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
