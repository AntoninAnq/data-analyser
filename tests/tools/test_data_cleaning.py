import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from tools.data_cleaning import (
    DataCleaningReport,
    save_cleaned_dataset,
    handle_missing_values,
    correct_data_types,
    normalize_data,
    remove_duplicates,
    remove_outliers
)

class TestDataCleaningReport:
    """Test the DataCleaningReport class"""
    
    def test_init(self):
        """Test initialization of DataCleaningReport"""
        report = DataCleaningReport()
        assert report.operations == []
        assert report.rows_modified == 0
        assert report.columns_modified == []
        assert report.original_shape is None
        assert report.final_shape is None
    
    def test_add_operation(self):
        """Test adding operations to the report"""
        report = DataCleaningReport()
        report.add_operation("Test Operation", "Test details", 10, ["col1", "col2"])
        
        assert len(report.operations) == 1
        assert report.operations[0]["type"] == "Test Operation"
        assert report.operations[0]["details"] == "Test details"
        assert report.operations[0]["rows_affected"] == 10
        assert report.operations[0]["columns_affected"] == ["col1", "col2"]
        assert report.rows_modified == 10
        assert "col1" in report.columns_modified
        assert "col2" in report.columns_modified
    
    def test_generate_markdown_report_empty(self):
        """Test markdown report generation with no operations"""
        report = DataCleaningReport()
        markdown = report.generate_markdown_report()
        assert "No cleaning operations were performed" in markdown
    
    def test_generate_markdown_report_with_operations(self):
        """Test markdown report generation with operations"""
        report = DataCleaningReport()
        report.original_shape = (100, 5)
        report.final_shape = (95, 5)
        report.add_operation("Test Operation", "Test details", 5, ["col1"])
        
        markdown = report.generate_markdown_report()
        assert "Data Cleaning Report" in markdown
        assert "Test Operation" in markdown
        assert "Test details" in markdown
        assert "5" in markdown  # rows affected

class TestSaveCleanedDataset:
    """Test the save_cleaned_dataset function"""
    
    def test_save_csv_dataset(self):
        """Test saving a cleaned CSV dataset"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        # Create a temporary CSV file to simulate original
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("A,B\n1,a\n2,b\n3,c\n")
            original_path = f.name
        
        try:
            cleaned_path = save_cleaned_dataset(df, original_path, "_test")
            
            # Check that file was created
            assert os.path.exists(cleaned_path)
            assert cleaned_path.endswith('.csv')
            
            # Check that data was saved correctly
            loaded_df = pd.read_csv(cleaned_path)
            pd.testing.assert_frame_equal(df, loaded_df)
            
        finally:
            # Cleanup
            if os.path.exists(original_path):
                os.unlink(original_path)
            if os.path.exists(cleaned_path):
                os.unlink(cleaned_path)
    
    def test_save_parquet_dataset(self):
        """Test saving a cleaned Parquet dataset"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        # Create a temporary parquet file to simulate original
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name)
            original_path = f.name
        
        try:
            cleaned_path = save_cleaned_dataset(df, original_path, "_test")
            
            # Check that file was created
            assert os.path.exists(cleaned_path)
            assert cleaned_path.endswith('.parquet')
            
            # Check that data was saved correctly
            loaded_df = pd.read_parquet(cleaned_path)
            pd.testing.assert_frame_equal(df, loaded_df)
            
        finally:
            # Cleanup
            if os.path.exists(original_path):
                os.unlink(original_path)
            if os.path.exists(cleaned_path):
                os.unlink(cleaned_path)

class TestHandleMissingValues:
    """Test the handle_missing_values function"""
    
    def test_handle_missing_values_file_not_found(self):
        """Test handling missing values with non-existent file"""
        result = handle_missing_values("nonexistent_file.csv", "mean")
        assert "Error" in result
        assert "not found" in result
    
    def test_handle_missing_values_no_missing(self, tmp_path):
        """Test handling missing values when there are none"""
        # Create test CSV file
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = handle_missing_values(str(file_path), "mean")
        assert "No missing values found" in result
    
    def test_handle_missing_values_with_mean(self, tmp_path):
        """Test handling missing values with mean strategy"""
        # Create test CSV file with missing values
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4],
            'B': ['a', 'b', np.nan, 'd']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = handle_missing_values(str(file_path), "mean")
        
        assert "Missing Values Handling" in result
        assert "mean" in result
        assert "Cleaned Dataset Saved" in result
        assert "Data Cleaning Report" in result
    
    def test_handle_missing_values_with_drop(self, tmp_path):
        """Test handling missing values with drop strategy"""
        # Create test CSV file with missing values
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4],
            'B': ['a', 'b', np.nan, 'd']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = handle_missing_values(str(file_path), "drop")
        
        assert "Missing Values Handling" in result
        assert "drop" in result
        assert "Cleaned Dataset Saved" in result
        assert "Data Cleaning Report" in result

class TestCorrectDataTypes:
    """Test the correct_data_types function"""
    
    def test_correct_data_types_file_not_found(self):
        """Test correcting data types with non-existent file"""
        result = correct_data_types("nonexistent_file.csv")
        assert "Error" in result
        assert "not found" in result
    
    def test_correct_data_types_no_object_columns(self, tmp_path):
        """Test correcting data types when there are no object columns"""
        # Create test CSV file with only numeric columns
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.5, 5.5, 6.5]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = correct_data_types(str(file_path))
        assert "No object columns found" in result
    
    def test_correct_data_types_with_conversion(self, tmp_path):
        """Test correcting data types with successful conversion"""
        # Create test CSV file with object columns that can be converted
        # Note: pandas will auto-convert '1', '2', '3' to int64 when reading CSV
        # So we need to use values that won't be auto-converted
        df = pd.DataFrame({
            'A': ['1.5a', '2.5b', '3.5c'],  # Mixed strings that can be cleaned to numeric
            'B': ['a', 'b', 'c']   # Should remain object
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = correct_data_types(str(file_path), force_numeric=True)
        
        assert "Data Type Correction" in result
        assert "Conversions Made" in result
        assert "Cleaned Dataset Saved" in result
        assert "Data Cleaning Report" in result

class TestNormalizeData:
    """Test the normalize_data function"""
    
    def test_normalize_data_file_not_found(self):
        """Test normalizing data with non-existent file"""
        result = normalize_data("nonexistent_file.csv")
        assert "Error" in result
        assert "not found" in result
    
    def test_normalize_data_no_numeric_columns(self, tmp_path):
        """Test normalizing data when there are no numeric columns"""
        # Create test CSV file with only object columns
        df = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['d', 'e', 'f']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = normalize_data(str(file_path))
        assert "No numerical columns found" in result
    
    def test_normalize_data_standard(self, tmp_path):
        """Test normalizing data with standard scaler"""
        # Create test CSV file with numeric columns
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = normalize_data(str(file_path), "standard")
        
        assert "Data Normalization" in result
        assert "StandardScaler" in result
        assert "Normalized Dataset Saved" in result
        assert "Data Cleaning Report" in result
    
    def test_normalize_data_robust(self, tmp_path):
        """Test normalizing data with robust scaler"""
        # Create test CSV file with numeric columns
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = normalize_data(str(file_path), "robust")
        
        assert "Data Normalization" in result
        assert "RobustScaler" in result
        assert "Normalized Dataset Saved" in result
        assert "Data Cleaning Report" in result
    
    def test_normalize_data_invalid_method(self, tmp_path):
        """Test normalizing data with invalid method"""
        # Create test CSV file with numeric columns
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = normalize_data(str(file_path), "invalid_method")
        assert "Error" in result
        assert "Unsupported normalization method" in result

class TestRemoveDuplicates:
    """Test the remove_duplicates function"""
    
    def test_remove_duplicates_file_not_found(self):
        """Test removing duplicates with non-existent file"""
        result = remove_duplicates("nonexistent_file.csv")
        assert "Error" in result
        assert "not found" in result
    
    def test_remove_duplicates_no_duplicates(self, tmp_path):
        """Test removing duplicates when there are none"""
        # Create test CSV file with no duplicates
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = remove_duplicates(str(file_path))
        assert "No duplicates found" in result
    
    def test_remove_duplicates_with_duplicates(self, tmp_path):
        """Test removing duplicates when there are duplicates"""
        # Create test CSV file with duplicates
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': ['a', 'b', 'b', 'c']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = remove_duplicates(str(file_path))
        
        assert "Duplicate Removal" in result
        assert "Duplicate Rows Found" in result
        assert "Deduplicated Dataset Saved" in result
        assert "Data Cleaning Report" in result

class TestRemoveOutliers:
    """Test the remove_outliers function"""
    
    def test_remove_outliers_file_not_found(self):
        """Test removing outliers with non-existent file"""
        result = remove_outliers("nonexistent_file.csv")
        assert "Error" in result
        assert "not found" in result
    
    def test_remove_outliers_no_numeric_columns(self, tmp_path):
        """Test removing outliers when there are no numeric columns"""
        # Create test CSV file with only object columns
        df = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['d', 'e', 'f']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = remove_outliers(str(file_path))
        assert "No numerical columns found" in result
    
    def test_remove_outliers_no_outliers(self, tmp_path):
        """Test removing outliers when there are none"""
        # Create test CSV file with no outliers
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = remove_outliers(str(file_path))
        assert "No outliers found" in result
    
    def test_remove_outliers_with_outliers(self, tmp_path):
        """Test removing outliers when there are outliers"""
        # Create test CSV file with outliers
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = remove_outliers(str(file_path))
        
        assert "Outlier Removal" in result
        assert "Outliers Found" in result
        assert "Cleaned Dataset Saved" in result
        assert "Data Cleaning Report" in result
    
    def test_remove_outliers_invalid_method(self, tmp_path):
        """Test removing outliers with invalid method"""
        # Create test CSV file with numeric columns
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        result = remove_outliers(str(file_path), method="invalid_method")
        assert "Error" in result
        assert "Unsupported outlier detection method" in result

class TestIntegration:
    """Integration tests for data cleaning workflow"""
    
    def test_complete_cleaning_workflow(self, tmp_path):
        """Test a complete data cleaning workflow"""
        # Create test CSV file with various issues
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 3, 4, 100],  # Missing value, duplicate, outlier
            'B': ['1a', '2b', '3c', '3d', '4e', '5f'],  # Object that can be converted to numeric with cleaning
            'C': ['a', 'b', 'c', 'c', 'd', 'e']   # Categorical with duplicate
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        # Test missing values handling
        result1 = handle_missing_values(str(file_path), "mean")
        assert "Missing Values Handling" in result1
        assert "Cleaned Dataset Saved" in result1
        
        # Test data type correction
        result2 = correct_data_types(str(file_path), force_numeric=True)
        assert "Data Type Correction" in result2
        assert "Cleaned Dataset Saved" in result2
        
        # Test duplicate removal (use subset to find duplicates in categorical column)
        result3 = remove_duplicates(str(file_path), subset="C")
        assert "Duplicate Removal" in result3
        assert "Deduplicated Dataset Saved" in result3
        
        # Test outlier removal
        result4 = remove_outliers(str(file_path))
        assert "Outlier Removal" in result4
        assert "Cleaned Dataset Saved" in result4

class TestSaveCurrentDataset:
    """Test the save_current_dataset functionality"""
    
    def test_save_current_dataset_no_dataset(self):
        """Test saving when no current dataset is available"""
        from tools.data_cleaning import save_current_dataset
        
        result = save_current_dataset()
        assert "Error" in result
        assert "No current dataset available" in result
    
    def test_save_current_dataset_with_dataset(self, tmp_path):
        """Test saving the current dataset"""
        from tools.data_cleaning import save_current_dataset
        from tools.dataset_manager import DatasetManager
        
        # Create test dataset
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        # Set up the dataset manager
        manager = DatasetManager()
        manager.set_current_dataset(df, str(file_path))
        manager.add_cleaning_operation({
            'type': 'Test Operation',
            'details': 'Test cleaning operation'
        })
        
        # Test saving
        result = save_current_dataset()
        
        assert "Dataset Saved Successfully" in result
        assert "File Information" in result
        assert "Cleaning History" in result
        assert "Test cleaning operation" in result
    
    def test_save_current_dataset_custom_path(self, tmp_path):
        """Test saving with custom path"""
        from tools.data_cleaning import save_current_dataset
        from tools.dataset_manager import DatasetManager
        
        # Create test dataset
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        # Set up the dataset manager
        manager = DatasetManager()
        manager.set_current_dataset(df, str(file_path))
        
        # Test saving with custom path
        custom_path = str(tmp_path / "custom_saved.csv")
        result = save_current_dataset(custom_path)
        
        assert "Dataset Saved Successfully" in result
        assert custom_path in result
        assert os.path.exists(custom_path)
