from typing import Dict
import pandas as pd
import numpy as np
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json
import re
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

class SFNModelTrainingAgent(SFNAgent):
    """Agent responsible for training a specific model with handling class imbalance"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Model Training", role="ML Engineer")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["model_trainer"]
        self.max_retries = 3  # Add max retries parameter
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """
        Train a specific model and return metrics and model object
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
        
        required_keys = ['df_train', 'df_valid', 'target_column', 'model_name']
        if not all(key in task.data for key in required_keys):
            raise ValueError(f"Task data must contain: {required_keys}")
        
        target_column = task.data.get('target_column')
        if not target_column:
            raise ValueError("Target column is not defined")
        if target_column not in task.data['df_train'].columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if last_error:
                    print(f">>>>>> Retry attempt {attempt + 1}/{self.max_retries} for model training. Previous error: {str(last_error)}")

                # Get training code from LLM with error feedback
                training_code, explanation = self._get_training_code(
                    task.data, 
                    previous_error=str(last_error) if last_error else None,
                    attempt=attempt + 1
                )

                # Add debug print for cleaned code
                print(">>> Generated Training Code:")
                print(training_code)
                print(">>> End Training Code")
                
                # Get target column from task data
                target_column = task.data.get('target_column')
                
                # Safely filter out records with null target values
                train_df = task.data['df_train'].copy()
                valid_df = task.data['df_valid'].copy()
                
                if target_column is None or target_column not in train_df.columns:
                    print(f"Warning: Target column '{target_column}' not found. Using all records.")
                    train_df_filtered = train_df
                    valid_df_filtered = valid_df
                else:
                    # Handle different types of null values
                    train_mask = (pd.notna(train_df[target_column]) & 
                                 (train_df[target_column] != '') & 
                                 (train_df[target_column] != 'None') & 
                                 (train_df[target_column].astype(str) != 'nan'))
                    valid_mask = (pd.notna(valid_df[target_column]) & 
                                 (valid_df[target_column] != '') & 
                                 (valid_df[target_column] != 'None') & 
                                 (valid_df[target_column].astype(str) != 'nan'))
                    
                    train_df_filtered = train_df[train_mask].copy()
                    valid_df_filtered = valid_df[valid_mask].copy()
                
                if len(train_df_filtered) == 0 or len(valid_df_filtered) == 0:
                    print("Warning: No valid records found after filtering. Using original data.")
                    train_df_filtered = train_df
                    valid_df_filtered = valid_df
                
                # Create namespace for execution
                globals_dict = {
                    'train_df': train_df_filtered,
                    'valid_df': valid_df_filtered,
                    'target_column': target_column,
                    'pd': pd,
                    'np': np,
                    'XGBClassifier': XGBClassifier,
                    'LGBMClassifier': LGBMClassifier,
                    'RandomForestClassifier': RandomForestClassifier,
                    'CatBoostClassifier': CatBoostClassifier,
                    'roc_auc_score': roc_auc_score,
                    'precision_score': precision_score,
                    'recall_score': recall_score,
                    'f1_score': f1_score,
                    'confusion_matrix': confusion_matrix
                }
                
                # Execute with globals_dict
                exec(training_code, globals_dict, globals_dict)
                
                print(f">>>>>> Successfully trained model on attempt {attempt + 1}")
                return {
                    'metrics': globals_dict.get('metrics', {}),
                    'model': globals_dict.get('model', None),
                    'training_features': globals_dict.get('training_features', []),
                    'records_info': {
                        'total_train': len(task.data['df_train']),
                        'used_train': len(train_df_filtered),
                        'total_valid': len(task.data['df_valid']),
                        'used_valid': len(valid_df_filtered)
                    }
                }

            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    print(f">>>>>> All {self.max_retries} attempts failed. Returning error response.")
                    return {
                        'metrics': {
                            'error': f'Training failed after {self.max_retries} attempts. Last error: {str(e)}',
                            'roc_auc': None,
                            'precision': None,
                            'recall': None,
                            'f1': None,
                            'confusion_matrix': None
                        },
                        'model': None,
                        'training_features': [],
                        'records_info': {
                            'total_train': len(task.data['df_train']),
                            'used_train': 0,
                            'total_valid': len(task.data['df_valid']),
                            'used_valid': 0
                        }
                    }

    def _get_training_code(self, data: Dict, previous_error=None, attempt=1) -> tuple[str, str]:
        """Get Python code for model training from LLM"""
        data_info = self._get_data_info(data)
        
        prompt_args = {
            'model_name': data['model_name'],
            'data_info': data_info,
            'target_column': data['target_column'],
            'date_column': data.get('date_column', 'billing_date'),
            'available_features': list(data['df_train'].columns),
            'categorical_features': [col for col in data['df_train'].select_dtypes(include=['object', 'category']).columns],
            'numeric_features': [col for col in data['df_train'].select_dtypes(include=['int64', 'float64']).columns]
        }
        
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='model_trainer',
            llm_provider=self.llm_provider,
            prompt_type='main',
            **prompt_args
        )

        if previous_error and attempt > 1:
            user_prompt += f"\n\nPrevious attempt {attempt-1} failed with error:\n{previous_error}\nPlease adjust the code to handle this error."

        response = self._get_llm_response(system_prompt, user_prompt)
        return self._parse_llm_response(response)

    def _get_data_info(self, data: Dict) -> Dict:
        """Gather information about the datasets with fallback to minimal info"""
        train_df = data['df_train']
        valid_df = data['df_valid']
        target_col = data.get('target_column')
        
        try:
            # Create copies to avoid modifying original data
            train_df = train_df.copy()
            valid_df = valid_df.copy()
            
            # Handle different types of null values in target column
            train_mask = pd.notna(train_df[target_col]) & (train_df[target_col] != '') & (train_df[target_col] != 'None')
            valid_mask = pd.notna(valid_df[target_col]) & (valid_df[target_col] != '') & (valid_df[target_col] != 'None')
            
            train_df_with_target = train_df[train_mask]
            valid_df_with_target = valid_df[valid_mask]
            
            # Safely compute target distribution with fallback
            try:
                train_dist = (train_df_with_target[target_col]
                             .fillna('Unknown')
                             .value_counts()
                             .to_dict())
                valid_dist = (valid_df_with_target[target_col]
                             .fillna('Unknown')
                             .value_counts()
                             .to_dict())
            except:
                # Fallback for target distribution
                train_dist = {'Unknown': len(train_df_with_target)}
                valid_dist = {'Unknown': len(valid_df_with_target)}
            
            # Get feature list excluding target column
            features = [col for col in train_df.columns if col != target_col]
            
            # Get dtypes safely with fallback
            try:
                dtypes_dict = {str(k): str(v) for k, v in train_df.dtypes.to_dict().items()}
            except:
                # Fallback for dtypes
                dtypes_dict = {col: 'unknown' for col in train_df.columns}
            
            return {
                'train_shape': train_df_with_target.shape,
                'valid_shape': valid_df_with_target.shape,
                'target_distribution_train': train_dist,
                'target_distribution_valid': valid_dist,
                'features': features,
                'dtypes': dtypes_dict,
                'column_mappings': {
                    'target_column': target_col,
                    'date_column': data.get('date_column', None),
                },
                'excluded_records': {
                    'train': len(train_df) - len(train_df_with_target),
                    'valid': len(valid_df) - len(valid_df_with_target)
                }
            }
        except Exception as e:
            print(f"Warning: Using minimal data info due to error: {str(e)}")
            # Return minimal data info that won't break the flow
            return {
                'train_shape': train_df.shape,
                'valid_shape': valid_df.shape,
                'target_distribution_train': {'Unknown': len(train_df)},
                'target_distribution_valid': {'Unknown': len(valid_df)},
                'features': [col for col in train_df.columns if col != target_col],
                'dtypes': {col: 'unknown' for col in train_df.columns},
                'column_mappings': {
                    'target_column': target_col,
                    'date_column': None
                },
                'excluded_records': {
                    'train': 0,
                    'valid': 0
                }
            }

    def _get_llm_response(self, system_prompt: str, user_prompt: str):
        """Get response from LLM"""
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.1,  # Low temperature for consistent code
            "max_tokens": 2000,
            "n": 1,
            "stop": None
        })
        
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": provider_config["temperature"],
            "max_tokens": provider_config["max_tokens"],
            "n": provider_config["n"],
            "stop": provider_config["stop"]
        }

        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )
        
        return response

    def _parse_llm_response(self, response) -> tuple[str, str]:
        """Parse LLM response into code and explanation"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response
            print(">>> content1", content)    
            # Clean markdown and get JSON content
            if "```" in content:
                # Extract content between code block markers
                parts = content.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        # Remove "json" if it's at the start
                        if part.startswith("json"):
                            part = part[4:]
                        content = part.strip()
                        break
            print(">>> content2", content)
            # Parse the JSON content
            response_dict = json.loads(content)
            print(">>> response_dict", response_dict)
            # Get code and fix indentation
            code = response_dict['code']
            # Remove any common leading whitespace from every line
            code_lines = code.splitlines()
            print(">>> code_lines", code_lines)
            if code_lines:
                # Find minimum indentation
                min_indent = float('inf')
                for line in code_lines:
                    if line.strip():  # Only check non-empty lines
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                print(">>> min_indent", min_indent)
                # Remove that amount of indentation from each line
                if min_indent < float('inf'):
                    code = '\n'.join(line[min_indent:] if line.strip() else ''
                                   for line in code_lines)
                print(">>> code", code)
            explanation = response_dict['explanation']
            print(">>> explanation", explanation)
            return code, explanation
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw content:\n{content}")
            raise ValueError("Failed to parse LLM response")

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
        
        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='model_trainer',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 