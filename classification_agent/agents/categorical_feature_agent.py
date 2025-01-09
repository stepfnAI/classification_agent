from typing import Dict
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import os

class SFNCategoricalFeatureAgent(SFNAgent):
    """Agent responsible for handling categorical feature encoding"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Categorical Feature Handler", role="Feature Engineer")
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["categorical_feature_handler"]
        
        # Initialize prompt manager
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict:
        """Process categorical features based on cardinality"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
        
        df = task.data['df']
        mappings = task.data['mappings']
        target_col = mappings.get('TARGET')
        
        # Make a copy to avoid modifying original
        modified_df = df.copy()
        
        # Identify categorical columns
        categorical_info = self._identify_categorical_columns(modified_df, mappings, target_col)
        
        # Apply encodings based on cardinality
        feature_info = {}
        
        for col, info in categorical_info.items():
            cardinality = info['cardinality']
            
            if cardinality <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(modified_df[col], prefix=col)
                modified_df = pd.concat([modified_df, dummies], axis=1)
                modified_df.drop(col, axis=1, inplace=True)
                feature_info[col] = {
                    'encoding_type': 'one_hot',
                    'cardinality': cardinality,
                    'new_columns': dummies.columns.tolist()
                }
                
            elif cardinality <= 50:
                # Label encoding for medium cardinality
                modified_df[col] = pd.Categorical(modified_df[col]).codes
                feature_info[col] = {
                    'encoding_type': 'label',
                    'cardinality': cardinality
                }
            
            else:
                # Ignore high cardinality
                feature_info[col] = {
                    'encoding_type': 'ignored',
                    'cardinality': cardinality,
                    'reason': 'high_cardinality'
                }
        
        return {
            'df': modified_df,
            'feature_info': feature_info
        }
    
    def _identify_categorical_columns(self, df: pd.DataFrame, mappings: Dict, target_col: str) -> Dict:
        """Identify categorical columns and their properties"""
        categorical_info = {}
        
        # Get all mapped fields that should be excluded from encoding
        mapped_fields = set(mappings.values())
        
        for col in df.columns:
            # Skip target column, mapped fields, and date/ID columns
            if (col == target_col or 
                col in mapped_fields or 
                col in [mappings.get('CUST_ID'), mappings.get('BILLING_DATE'), 
                       mappings.get('PRODUCT_ID'), mappings.get('TARGET')]):
                continue
            
            # Check if column is categorical
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                cardinality = df[col].nunique()
                categorical_info[col] = {
                    'cardinality': cardinality,
                    'sample_values': df[col].value_counts().head(5).to_dict()
                }
        
        return categorical_info
    
    def get_validation_params(self):
        """Return parameters for validation"""
        return {
            'required_fields': ['df', 'mappings'],
            'min_features': 1
        } 