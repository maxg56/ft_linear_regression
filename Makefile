# Makefile for ft_linear_regression project

.PHONY: help train  estimate demo clean install 

# Default target
help:
	@echo "ft_linear_regression - Linear Regression from Scratch"
	@echo ""
	@echo "Available targets:"
	@echo "  help       - Show this help message"
	@echo "  install    - Install required dependencies"
	@echo "  train      - Train linear regression model"
	@echo "  demo       - Complete demonstration (data + regression + precision)"
	@echo "  estimate   - Interactive price estimation (run 'make estimate KM=<value>')"
	@echo "  clean      - Clean generated files"
	@echo "  all        - Train model and generate all visualizations"
	@echo ""
	@echo "Examples:"
	@echo "  make train"
	@echo "  make demo"
	@echo "  make estimate KM=50000"

# Install dependencies
install:
	@echo "Installing required dependencies..."
	if [ ! -d "venv" ]; then \
		python3 -m venv venv; \
	fi
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	@echo "Training linear regression model..."
	cd src && python linear_regression.py

demo:
	@echo "Running complete demonstration..."
	cd src && python demo.py

estimate:
	@echo "Estimating price for $(KM) km..."
	cd src && python estimatePrice.py $(KM)

clean:
	@echo "Cleaning generated files..."
	rm -rf src/__pycache__
	rm -f graphs/*.png
	@echo "Cleaned!"

