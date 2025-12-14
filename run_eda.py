#!/usr/bin/env python3
"""
Script to run EDA analysis on the credit risk data.
This is a Python script version of the Jupyter notebook for execution.
"""

import os
import warnings
from pathlib import Path
import sys

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
from scipy.stats import skew, kurtosis

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - Credit Risk Model")
print("=" * 80)

# Define data paths
project_root = Path(__file__).resolve().parent
data_raw_path = project_root / 'data' / 'raw'
output_path = project_root / 'notebooks' / 'eda_outputs'
output_path.mkdir(exist_ok=True)

print(f"\nProject root: {project_root}")
print(f"Data path: {data_raw_path}")

# Load data
if not data_raw_path.exists():
    print(f"\nERROR: Data directory not found at {data_raw_path}")
    sys.exit(1)

data_files = list(data_raw_path.glob('*.csv'))
print(f"\nFound {len(data_files)} CSV file(s) in data/raw/")
for file in data_files:
    print(f"  - {file.name}")

# Load the main transaction data
data_file = data_raw_path / 'data.csv'
if data_file.exists():
    print(f"\nLoading data from: {data_file.name}")
    try:
        df = pd.read_csv(data_file, low_memory=False)
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
else:
    print(f"\nERROR: data.csv not found in {data_raw_path}")
    sys.exit(1)

# Data Overview
print("\n" + "=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 80)
print("DATA TYPES AND MISSING VALUES")
print("=" * 80)
dtype_info = pd.DataFrame({
    'Column': df.dtypes.index,
    'Data Type': df.dtypes.values,
    'Non-Null Count': df.count().values,
    'Null Count': df.isnull().sum().values,
    'Null Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
})
print(dtype_info.to_string())

# Identify column types
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Summary Statistics
if numerical_cols:
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - NUMERICAL FEATURES")
    print("=" * 80)
    print(df[numerical_cols].describe().to_string())
    
    # Additional statistics
    additional_stats = pd.DataFrame({
        'Column': numerical_cols,
        'Skewness': [skew(df[col].dropna()) for col in numerical_cols],
        'Kurtosis': [kurtosis(df[col].dropna()) for col in numerical_cols],
        'Min': [df[col].min() for col in numerical_cols],
        'Max': [df[col].max() for col in numerical_cols],
        'Range': [df[col].max() - df[col].min() for col in numerical_cols],
        'IQR': [df[col].quantile(0.75) - df[col].quantile(0.25) for col in numerical_cols]
    })
    print("\nAdditional Statistics:")
    print(additional_stats.to_string())

# Categorical summary
if categorical_cols:
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES SUMMARY")
    print("=" * 80)
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Top 10 values:")
        top_values = df[col].value_counts().head(10)
        for val, count in top_values.items():
            print(f"    {val}: {count:,} ({count/len(df)*100:.2f}%)")

# Missing Values Analysis
print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)
missing_data = dtype_info[dtype_info['Null Count'] > 0].sort_values('Null Count', ascending=False)
if len(missing_data) > 0:
    print(f"\nColumns with missing values: {len(missing_data)}")
    print(missing_data[['Column', 'Null Count', 'Null Percentage']].to_string())
else:
    print("\n✓ No missing values found in the dataset!")

# Correlation Analysis
if len(numerical_cols) > 1:
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    corr_matrix = df[numerical_cols].corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if high_corr_pairs:
        print("\nHigh Correlations (|r| > 0.7):")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.sort_values('Correlation', key=abs, ascending=False).to_string())
    else:
        print("\nNo high correlations found (|r| > 0.7)")

# Outlier Detection
if numerical_cols:
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION (IQR METHOD)")
    print("=" * 80)
    outlier_summary = []
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_summary.append({
            'Column': col,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outlier Count': outlier_count,
            'Outlier Percentage': outlier_percentage
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.sort_values('Outlier Percentage', ascending=False).to_string())

# Time Series Analysis
if 'TransactionStartTime' in df.columns:
    print("\n" + "=" * 80)
    print("TIME SERIES ANALYSIS")
    print("=" * 80)
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Year'] = df['TransactionStartTime'].dt.year
        df['Month'] = df['TransactionStartTime'].dt.month
        df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        
        print(f"\nDate Range: {df['TransactionStartTime'].min()} to {df['TransactionStartTime'].max()}")
        print(f"Total Days: {(df['TransactionStartTime'].max() - df['TransactionStartTime'].min()).days}")
        
        daily_transactions = df.groupby(df['TransactionStartTime'].dt.date).size()
        print(f"\nDaily Transaction Statistics:")
        print(f"  Mean: {daily_transactions.mean():.2f}")
        print(f"  Median: {daily_transactions.median():.2f}")
        print(f"  Std: {daily_transactions.std():.2f}")
        print(f"  Min: {daily_transactions.min()}")
        print(f"  Max: {daily_transactions.max()}")
        
    except Exception as e:
        print(f"Error in time series analysis: {e}")

# Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Distribution plots for numerical features
if numerical_cols:
    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        if idx < len(axes):
            ax = axes[idx]
            df[col].dropna().hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'numerical_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: numerical_distributions.png")

# Correlation heatmap
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: correlation_heatmap.png")

# Summary
print("\n" + "=" * 80)
print("EDA SUMMARY")
print("=" * 80)
summary_stats = {
    'Total Rows': len(df),
    'Total Columns': len(df.columns),
    'Numerical Features': len(numerical_cols),
    'Categorical Features': len(categorical_cols),
    'Missing Values Total': df.isnull().sum().sum(),
    'Duplicate Rows': df.duplicated().sum()
}

for key, value in summary_stats.items():
    print(f"{key}: {value:,}")

print("\n" + "=" * 80)
print("EDA COMPLETED SUCCESSFULLY!")
print(f"Outputs saved to: {output_path}")
print("=" * 80)

