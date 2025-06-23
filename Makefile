# Makefile for ft_linear_regression project

.PHONY: help train test estimate visualize clean install all

# Default target
help:
	@echo "ft_linear_regression - Linear Regression from Scratch"
	@echo ""
	@echo "Available targets:"
	@echo "  help       - Show this help message"
	@echo "  install    - Install required dependencies"
	@echo "  train      - Train the linear regression model"
	@echo "  test       - Test the trained model"
	@echo "  visualize  - Generate all visualization plots"
	@echo "  estimate   - Interactive price estimation (run 'make estimate KM=<value>')"
	@echo "  clean      - Clean generated files"
	@echo "  all        - Train model and generate visualizations"
	@echo ""
	@echo "Examples:"
	@echo "  make train"
	@echo "  make estimate KM=50000"
	@echo "  make all"

# Install dependencies
install:
	@echo "Installing required dependencies..."
	pip install numpy pandas matplotlib

# Train the model
train:
	@echo "Training linear regression model..."
	cd src && python train.py

# Test the model
test:
	@echo "Testing the trained model..."
	cd src && python test.py

# Generate visualizations
visualize:
	@echo "Generating visualization plots..."
	cd src && python graf.py

# Estimate price for specific mileage
estimate:
	@echo "Estimating price for $(KM) km..."
	cd src && python estimatePrice.py $(KM)

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf src/__pycache__
	rm -f graphs/*.png
	@echo "Cleaned!"

# Train and visualize
all: train visualize
	@echo "Model training and visualization completed!"

# Run core linear regression
core:
	@echo "Running core linear regression..."
	cd src && python linear_regression.py
