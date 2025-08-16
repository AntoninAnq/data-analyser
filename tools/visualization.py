from crewai.tools import tool
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
import os
from datetime import datetime
from .utils import load_dataset, validate_column_exists
import pandas as pd

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

@tool("Create a distribution plot for a specific column.")
def distribution_plot_tool(file_path: str, column_name: str) -> str:
    """
    Tool for creating a distribution plot for a specific column.
    
    Args:
        file_path (str): Path to the dataset file
        column_name (str): Name of the column to plot
    """
    return distribution_plot(file_path, column_name)

def distribution_plot(file_path: str, column_name: str) -> str:
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
        
        # Check if column has sufficient variation
        if column_data.nunique() <= 1:
            return f"❌ **Error**: Column '{column_name}' has no variation (all values are the same). Cannot create a meaningful distribution plot."
        
        # Check if column is numeric for distribution analysis
        if not pd.api.types.is_numeric_dtype(column_data):
            return f"❌ **Error**: Column '{column_name}' is not numeric. For categorical data, use a bar plot instead."
        
        # Check for sufficient data points
        if len(column_data) < 3:
            return f"❌ **Error**: Insufficient data in column '{column_name}' for distribution analysis (need at least 3 data points)."
        
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

@tool("Create a correlation heatmap for numerical columns.")
def correlation_heatmap_tool(file_path: str, columns: Optional[str] = None) -> str:
    """
    Tool for creating a correlation heatmap for numerical columns.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str, optional): Comma-separated list of columns to include. If None, uses all numerical columns.
    """
    return correlation_heatmap(file_path, columns)

def correlation_heatmap(file_path: str, columns: Optional[str] = None) -> str:
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
        
        # Check for sufficient data variation
        if len(numerical_df) < 3:
            return "❌ **Error**: Insufficient data for correlation analysis (need at least 3 rows)."
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Check if correlation matrix is valid (not all NaN)
        if corr_matrix.isnull().all().all():
            return "❌ **Error**: Cannot calculate correlations - all values are NaN or constant."
        
        # Check if correlation matrix has any valid values
        if corr_matrix.isnull().all().all() or (corr_matrix == 0).all().all():
            return "❌ **Error**: No meaningful correlations found - all correlations are zero or NaN."
        
        # Check for constant columns (which cause NaN correlations)
        constant_columns = []
        for col in numerical_df.columns:
            if numerical_df[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            return f"❌ **Error**: Cannot create correlation heatmap. The following columns have no variation (constant values): {', '.join(constant_columns)}"
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Use a clean correlation matrix (handle any remaining NaN values)
        corr_matrix_clean = corr_matrix.fillna(0)
        
        sns.heatmap(corr_matrix_clean, 
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
        
        # Find strongest correlations (excluding NaN and zero values)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value) and corr_value != 0:
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        if not corr_pairs:
            return "❌ **Error**: No meaningful correlations found between the numerical columns."
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_correlations = corr_pairs[:5]
        
        correlation_text = "\n**Top correlations:**\n"
        for col1, col2, corr in top_correlations:
            correlation_text += f"- {col1} ↔ {col2}: {corr:.3f}\n"
        
        return f"✅ **Correlation heatmap created successfully!**\n\n**File saved as:** {filepath}\n\n**Analysis includes:** {len(numerical_df.columns)} numerical columns\n{correlation_text}"
        
    except Exception as e:
        return f"❌ **Error creating correlation heatmap**: {e}"

@tool("Create a pair plot for numerical columns.")
def pair_plot_tool(file_path: str, columns: Optional[str] = None, sample_size: int = 1000) -> str:
    """
    Create a pair plot for numerical columns.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str, optional): Comma-separated list of columns to include. If None, uses all numerical columns.
        sample_size (int): Number of samples to use for plotting (to avoid overcrowding)
    """
    return pair_plot(file_path, columns, sample_size)

def pair_plot(file_path: str, columns: Optional[str] = None, sample_size: int = 1000) -> str:
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

@tool("Create a scatter plot between two columns.")
def scatter_plot_tool(file_path: str, x_column: str, y_column: str, color_column: Optional[str] = None) -> str:
    """
    Tool for creating a scatter plot between two columns.
    
    Args:
        file_path (str): Path to the dataset file
        x_column (str): Name of the column for x-axis
        y_column (str): Name of the column for y-axis
        color_column (str, optional): Name of the column to use for color coding
    """
    return scatter_plot(file_path, x_column, y_column, color_column)

def scatter_plot(file_path: str, x_column: str, y_column: str, color_column: Optional[str] = None) -> str:
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
        
        # Check if columns are numeric
        for col in [x_column, y_column]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return f"❌ **Error**: Column '{col}' is not numeric. Scatter plots require numerical data."
        
        # Get data
        plot_data = df[[x_column, y_column]].dropna()
        if color_column:
            plot_data[color_column] = df[color_column]
            plot_data = plot_data.dropna()
        
        if len(plot_data) == 0:
            return "❌ **Error**: No data available after removing missing values."
        
        # Check for sufficient data points
        if len(plot_data) < 3:
            return "❌ **Error**: Insufficient data for scatter plot (need at least 3 data points)."
        
        # Check for variation in data
        if plot_data[x_column].nunique() <= 1:
            return f"❌ **Error**: Column '{x_column}' has no variation (all values are the same). Cannot create a meaningful scatter plot."
        
        if plot_data[y_column].nunique() <= 1:
            return f"❌ **Error**: Column '{y_column}' has no variation (all values are the same). Cannot create a meaningful scatter plot."
        
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
        
        # Add trend line (only if we have enough data points)
        if len(plot_data) >= 3:
            try:
                z = np.polyfit(plot_data[x_column], plot_data[y_column], 1)
                p = np.poly1d(z)
                plt.plot(plot_data[x_column], p(plot_data[x_column]), "r--", alpha=0.8, linewidth=2)
            except:
                # If trend line fails, continue without it
                pass
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = create_plot_directory()
        filepath = save_plot(plt.gcf(), f"scatter_{x_column}_vs_{y_column}", plot_dir)
        
        # Calculate correlation
        correlation = plot_data[x_column].corr(plot_data[y_column])
        
        return f"✅ **Scatter plot created successfully!**\n\n**File saved as:** {filepath}\n\n**Plot shows:**\n- Relationship between {x_column} and {y_column}\n- Red dashed line shows the trend\n- Correlation coefficient: {correlation:.3f}\n\n**Data points:** {len(plot_data):,}"
        
    except Exception as e:
        return f"❌ **Error creating scatter plot**: {e}"

@tool("Create a bar plot for categorical or discrete numerical columns.")
def bar_plot_tool(file_path: str, column_name: str, top_n: int = 10) -> str:
    """
    Tool for creating a bar plot for categorical or discrete numerical columns.
    
    Args:
        file_path (str): Path to the dataset file
        column_name (str): Name of the column to plot
        top_n (int): Number of top categories to show
    """
    return bar_plot(file_path, column_name, top_n)

def bar_plot(file_path: str, column_name: str, top_n: int = 10) -> str:
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
        
        # Get column data
        column_data = df[column_name].dropna()
        
        if len(column_data) == 0:
            return f"❌ **Error**: Column '{column_name}' contains only missing values."
        
        # Check if column has sufficient variation
        if column_data.nunique() <= 1:
            return f"❌ **Error**: Column '{column_name}' has no variation (all values are the same). Cannot create a meaningful bar plot."
        
        # Check for sufficient data points
        if len(column_data) < 2:
            return f"❌ **Error**: Insufficient data in column '{column_name}' for bar plot (need at least 2 data points)."
        
        # Get value counts
        value_counts = column_data.value_counts().head(top_n)
        
        if len(value_counts) == 0:
            return f"❌ **Error**: No valid data found in column '{column_name}' after processing."
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
        plt.title(f'Top {len(value_counts)} Values in {column_name}', fontsize=14)
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
        
        return f"✅ **Bar plot created successfully!**\n\n**File saved as:** {filepath}\n\n**Plot shows:**\n- Top {len(value_counts)} most frequent values in {column_name}\n- Total unique values: {column_data.nunique()}\n- Missing values: {df[column_name].isnull().sum():,}"
        
    except Exception as e:
        return f"❌ **Error creating bar plot**: {e}"
