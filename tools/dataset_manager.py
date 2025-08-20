import pandas as pd
import threading
from datetime import datetime
from typing import Optional, Dict, List
from .utils import load_dataset

class DatasetManager:
    """Singleton class to manage the current working dataset across all agents and operations"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatasetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.current_dataset = None
            self.current_file_path = None
            self.original_file_path = None
            self.cleaning_history = []
            self.operation_history = []
            self._initialized = True
    
    def set_current_dataset(self, df: pd.DataFrame, file_path: str, original_file_path: str = None):
        """Set the current working dataset"""
        self.current_dataset = df.copy()
        self.current_file_path = file_path
        if original_file_path:
            self.original_file_path = original_file_path
        elif not self.original_file_path:
            self.original_file_path = file_path
    
    def get_current_dataset(self) -> pd.DataFrame:
        """Get the current working dataset"""
        return self.current_dataset.copy() if self.current_dataset is not None else None
    
    def get_current_file_path(self) -> str:
        """Get the current file path"""
        return self.current_file_path
    
    def get_original_file_path(self) -> str:
        """Get the original file path"""
        return self.original_file_path
    
    def add_cleaning_operation(self, operation: dict):
        """Add a cleaning operation to the history"""
        operation['timestamp'] = datetime.now().isoformat()
        operation['type'] = 'cleaning'
        self.cleaning_history.append(operation)
        self.operation_history.append(operation)
    
    def add_operation(self, operation: dict):
        """Add any operation to the general history"""
        operation['timestamp'] = datetime.now().isoformat()
        self.operation_history.append(operation)
    
    def get_cleaning_history(self) -> list:
        """Get the cleaning history"""
        return self.cleaning_history.copy()
    
    def get_operation_history(self) -> list:
        """Get all operation history"""
        return self.operation_history.copy()
    
    def clear_history(self):
        """Clear all history"""
        self.cleaning_history = []
        self.operation_history = []
    
    def reset_to_original(self):
        """Reset to the original dataset"""
        if self.original_file_path and self.original_file_path != self.current_file_path:
            result = load_dataset(self.original_file_path)
            if isinstance(result, pd.DataFrame):
                self.current_dataset = result
                self.current_file_path = self.original_file_path
                self.clear_history()
    
    def has_current_dataset(self) -> bool:
        """Check if there's a current dataset available"""
        return self.current_dataset is not None
    
    def get_dataset_info(self) -> dict:
        """Get information about the current dataset"""
        if self.current_dataset is None:
            return {}
        
        return {
            'shape': self.current_dataset.shape,
            'columns': list(self.current_dataset.columns),
            'dtypes': self.current_dataset.dtypes.to_dict(),
            'missing_counts': self.current_dataset.isnull().sum().to_dict(),
            'current_file_path': self.current_file_path,
            'original_file_path': self.original_file_path,
            'cleaning_operations_count': len(self.cleaning_history),
            'total_operations_count': len(self.operation_history)
        }
