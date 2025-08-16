import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Union, List, Optional
import os
from datetime import datetime
from .utils import load_dataset, validate_column_exists

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_plot_directory() -> str:
    """Create a directory for storing plots if it doesn't exist."""
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return plot_dir

def save_plot(plot, filename: str, plot_dir: str = "plots") -> str:
    """Save a plot and return the file path."""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(plot_dir, f"{filename}_{timestamp}.png")
    
    # Save with high DPI for better quality
    plot.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(plot)
    
    return filepath

def distribution_plot_tool(file_path: str, column_name: str) -> str:
    """
    Create a distribution plot for a specific column.
    
    Args:
        file_path (str): Path to the dataset file
        column_name (str): Name of the column to plot
        
    Returns:
        str: Path to the saved plot file or error message
    """
    try:
        # Load dataset
        df = load_dataset(file_path)
        if isinstance(df, str):
            return df
        
        # Validate column exists
        column_check = validate_column_exists(df, column_name)
        if isinstance(column_check, str):
            return column_check
        
        # Get column data
        column_data = df[column_name].dropna()
        
        if len(column_data) == 0:
            return f"❌ **Error**: Column '{column_name}' contains only missing values."
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(column_data, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Distribution of {column_name}')
        ax1.set_xlabel(column_name)
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(column_data, vert=False)
        ax2.set_title(f'Box Plot of {column_name}')
        ax2.set_xlabel(column_name)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = create_plot_directory()
        filepath = save_plot(plt.gcf(), f"distribution_{column_name}", plot_dir)
        
        return f"✅ **Distribution plot created successfully!**\n\n**File saved as:** {filepath}\n\n**Plot shows:**\n- Histogram showing the frequency distribution of values\n- Box plot showing median, quartiles, and outliers\n\n**Column statistics:**\n- Count: {len(column_data):,}\n- Mean: {column_data.mean():.2f}\n- Median: {column_data.median():.2f}\n- Std: {column_data.std():.2f}\n- Min: {column_data.min():.2f}\n- Max: {column_data.max():.2f}"
        
    except Exception as e:
        return f"❌ **Error creating distribution plot**: {e}"

def correlation_heatmap_tool(file_path: str, columns: Optional[str] = None) -> str:
    """
    Create a correlation heatmap for numerical columns.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str, optional): Comma-separated list of columns to include. If None, uses all numerical columns.
        
    Returns:
        str: Path to the saved plot file or error message
    """
    try:
        # Load dataset
        df = load_dataset(file_path)
        if isinstance(df, str):
            return df
        
        # Select numerical columns
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            # Validate all columns exist
            for col in column_list:
                column_check = validate_column_exists(df, col)
                if isinstance(column_check, str):
                    return column_check
            numerical_df = df[column_list].select_dtypes(include=[np.number])
        else:
            numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            return "❌ **Error**: No numerical columns found in the dataset."
        
        if len(numerical_df.columns) < 2:
            return "❌ **Error**: At least 2 numerical columns are required for correlation analysis."
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save plot
        plot_dir = create_plot_directory()
        filepath = save_plot(plt.gcf(), "correlation_heatmap", plot_dir)
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_correlations = corr_pairs[:5]
        
        correlation_text = "\n**Top correlations:**\n"
        for col1, col2, corr in top_correlations:
            correlation_text += f"- {col1} ↔ {col2}: {corr:.3f}\n"
        
        return f"✅ **Correlation heatmap created successfully!**\n\n**File saved as:** {filepath}\n\n**Analysis includes:** {len(numerical_df.columns)} numerical columns\n{correlation_text}"
        
    except Exception as e:
        return f"❌ **Error creating correlation heatmap**: {e}"

def pair_plot_tool(file_path: str, columns: Optional[str] = None, sample_size: int = 1000) -> str:
    """
    Create a pair plot for numerical columns.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str, optional): Comma-separated list of columns to include. If None, uses all numerical columns.
        sample_size (int): Number of samples to use for plotting (to avoid overcrowding)
        
    Returns:
        str: Path to the saved plot file or error message
    """
    try:
        # Load dataset
        df = load_dataset(file_path)
        if isinstance(df, str):
            return df
        
        # Select numerical columns
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            # Validate all columns exist
            for col in column_list:
                column_check = validate_column_exists(df, col)
                if isinstance(column_check, str):
                    return column_check
            numerical_df = df[column_list].select_dtypes(include=[np.number])
        else:
            numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            return "❌ **Error**: No numerical columns found in the dataset."
        
        if len(numerical_df.columns) < 2:
            return "❌ **Error**: At least 2 numerical columns are required for pair plot."
        
        # Limit number of columns to avoid overcrowded plots
        if len(numerical_df.columns) > 6:
            numerical_df = numerical_df.iloc[:, :6]
        
        # Sample data if too large
        if len(numerical_df) > sample_size:
            numerical_df = numerical_df.sample(n=sample_size, random_state=42)
        
        # Create pair plot
        plt.figure(figsize=(15, 15))
        pair_plot = sns.pairplot(numerical_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        pair_plot.fig.suptitle('Pair Plot of Numerical Variables', y=1.02, fontsize=16)
        pair_plot.fig.tight_layout()
        
        # Save plot
        plot_dir = create_plot_directory()
        filepath = save_plot(pair_plot.fig, "pair_plot", plot_dir)
        
        return f"✅ **Pair plot created successfully!**\n\n**File saved as:** {filepath}\n\n**Analysis includes:** {len(numerical_df.columns)} numerical columns\n**Sample size:** {len(numerical_df):,} rows\n\n**Plot shows:**\n- Scatter plots for all pairs of variables\n- Kernel density estimation (KDE) plots on the diagonal\n- Relationships and patterns between variables"
        
    except Exception as e:
        return f"❌ **Error creating pair plot**: {e}"

def scatter_plot_tool(file_path: str, x_column: str, y_column: str, color_column: Optional[str] = None) -> str:
    """
    Create a scatter plot between two columns.
    
    Args:
        file_path (str): Path to the dataset file
        x_column (str): Name of the column for x-axis
        y_column (str): Name of the column for y-axis
        color_column (str, optional): Name of the column to use for color coding
        
    Returns:
        str: Path to the saved plot file or error message
    """
    try:
        # Load dataset
        df = load_dataset(file_path)
        if isinstance(df, str):
            return df
        
        # Validate columns exist
        for col in [x_column, y_column]:
            column_check = validate_column_exists(df, col)
            if isinstance(column_check, str):
                return column_check
        
        if color_column:
            column_check = validate_column_exists(df, color_column)
            if isinstance(column_check, str):
                return column_check
        
        # Get data
        plot_data = df[[x_column, y_column]].dropna()
        if color_column:
            plot_data[color_column] = df[color_column]
            plot_data = plot_data.dropna()
        
        if len(plot_data) == 0:
            return "❌ **Error**: No data available after removing missing values."
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        if color_column:
            scatter = plt.scatter(plot_data[x_column], plot_data[y_column], 
                                c=plot_data[color_column], alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, label=color_column)
            title = f'Scatter Plot: {x_column} vs {y_column} (colored by {color_column})'
        else:
            plt.scatter(plot_data[x_column], plot_data[y_column], alpha=0.6)
            title = f'Scatter Plot: {x_column} vs {y_column}'
        
        plt.title(title, fontsize=14)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(plot_data[x_column], plot_data[y_column], 1)
        p = np.poly1d(z)
        plt.plot(plot_data[x_column], p(plot_data[x_column]), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = create_plot_directory()
        filepath = save_plot(plt.gcf(), f"scatter_{x_column}_vs_{y_column}", plot_dir)
        
        # Calculate correlation
        correlation = plot_data[x_column].corr(plot_data[y_column])
        
        return f"✅ **Scatter plot created successfully!**\n\n**File saved as:** {filepath}\n\n**Plot shows:**\n- Relationship between {x_column} and {y_column}\n- Red dashed line shows the trend\n- Correlation coefficient: {correlation:.3f}\n\n**Data points:** {len(plot_data):,}"
        
    except Exception as e:
        return f"❌ **Error creating scatter plot**: {e}"

def bar_plot_tool(file_path: str, column_name: str, top_n: int = 10) -> str:
    """
    Create a bar plot for categorical or discrete numerical columns.
    
    Args:
        file_path (str): Path to the dataset file
        column_name (str): Name of the column to plot
        top_n (int): Number of top categories to show
        
    Returns:
        str: Path to the saved plot file or error message
    """
    try:
        # Load dataset
        df = load_dataset(file_path)
        if isinstance(df, str):
            return df
        
        # Validate column exists
        column_check = validate_column_exists(df, column_name)
        if isinstance(column_check, str):
            return column_check
        
        # Get value counts
        value_counts = df[column_name].value_counts().head(top_n)
        
        if len(value_counts) == 0:
            return f"❌ **Error**: Column '{column_name}' contains only missing values."
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
        plt.title(f'Top {top_n} Values in {column_name}', fontsize=14)
        plt.xlabel(column_name, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = create_plot_directory()
        filepath = save_plot(plt.gcf(), f"bar_plot_{column_name}", plot_dir)
        
        return f"✅ **Bar plot created successfully!**\n\n**File saved as:** {filepath}\n\n**Plot shows:**\n- Top {top_n} most frequent values in {column_name}\n- Total unique values: {df[column_name].nunique()}\n- Missing values: {df[column_name].isnull().sum():,}"
        
    except Exception as e:
        return f"❌ **Error creating bar plot**: {e}"
