from typing import Dict
import pandas as pd
import numpy as np
from sfn_blueprint import SFNAgent, Task

class SFNLeakageDetectionAgent(SFNAgent):
    """Agent responsible for detecting potential target leakage in features"""
    
    def __init__(self):
        super().__init__(name="Leakage Detection", role="Data Validator")
        # Thresholds for leakage detection
        self.SEVERE_CORRELATION_THRESHOLD = 0.95  # Definite leakage
        self.HIGH_CORRELATION_THRESHOLD = 0.85    # Suspicious, needs review
        self.HIGH_NULL_PERCENTAGE = 0.30          # High missing values threshold
        self.HIGH_CARDINALITY_RATIO = 0.90       # High unique values ratio
        
    def execute_task(self, task: Task) -> Dict:
        """Analyze features for potential target leakage"""
        if not isinstance(task.data.get('df'), pd.DataFrame):
            raise ValueError("Task data must contain a pandas DataFrame under 'df' key")

        df = task.data['df']
        target_col = task.data['target_column']
        
        return self._analyze_leakage(df, target_col)

    def _analyze_leakage(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze features for leakage based on correlation, nulls, and cardinality"""
        findings = {
            'severe_leakage': [],      # Features with correlation > 0.95
            'suspicious_leakage': [],   # Features with correlation 0.90-0.95
            'analysis': {},            # Detailed analysis per feature
            'recommendations': {
                'remove': [],          # Features to definitely remove
                'review': []           # Features to review
            }
        }
        
        for column in df.columns:
            if column == target_col:
                continue
                
            analysis = {}
            
            # 1. Correlation Check
            correlation = self._calculate_correlation(df[column], df[target_col])
            analysis['correlation'] = correlation
            
            # 2. Missing Values Check
            null_percentage = df[column].isnull().mean()
            analysis['null_percentage'] = null_percentage
            
            # 3. Cardinality Check
            unique_ratio = df[column].nunique() / len(df)
            analysis['unique_ratio'] = unique_ratio
            
            # Store detailed analysis
            findings['analysis'][column] = analysis
            
            # Evaluate leakage based on all factors
            if abs(correlation) > self.SEVERE_CORRELATION_THRESHOLD:
                findings['severe_leakage'].append(column)
                findings['recommendations']['remove'].append({
                    'feature': column,
                    'reason': f"Very high correlation ({correlation:.3f}) with target"
                })
            
            elif abs(correlation) > self.HIGH_CORRELATION_THRESHOLD:
                findings['suspicious_leakage'].append(column)
                findings['recommendations']['review'].append({
                    'feature': column,
                    'reason': (
                        f"High correlation ({correlation:.3f}) with target. "
                        f"Null%: {null_percentage:.1%}, "
                        f"Unique ratio: {unique_ratio:.2f}"
                    )
                })
            
            # Additional suspicious patterns
            elif (null_percentage > self.HIGH_NULL_PERCENTAGE and 
                  abs(correlation) > 0.85):  # High correlation despite many nulls
                findings['recommendations']['review'].append({
                    'feature': column,
                    'reason': (
                        f"High correlation ({correlation:.3f}) despite "
                        f"high null percentage ({null_percentage:.1%})"
                    )
                })
            
            elif (unique_ratio > self.HIGH_CARDINALITY_RATIO and 
                  abs(correlation) > 0.85):  # High correlation with high cardinality
                findings['recommendations']['review'].append({
                    'feature': column,
                    'reason': (
                        f"High correlation ({correlation:.3f}) with "
                        f"high cardinality ratio ({unique_ratio:.2f})"
                    )
                })
        
        return findings

    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation between two series handling different data types"""
        try:
            # For numeric data
            if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
                return series1.corr(series2)
            
            # For categorical data, convert to numeric if possible
            elif pd.api.types.is_categorical_dtype(series1) or pd.api.types.is_categorical_dtype(series2):
                try:
                    return pd.to_numeric(series1, errors='coerce').corr(
                        pd.to_numeric(series2, errors='coerce')
                    )
                except:
                    return 0.0
            
            # For other types, try to convert to numeric
            else:
                try:
                    return pd.to_numeric(series1, errors='coerce').corr(
                        pd.to_numeric(series2, errors='coerce')
                    )
                except:
                    return 0.0
                    
        except Exception as e:
            print(f"Error calculating correlation: {str(e)}")
            return 0.0 