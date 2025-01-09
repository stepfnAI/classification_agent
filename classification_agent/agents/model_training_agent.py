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
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """
        Train a specific model and return metrics and model object
        
        :param task: Task object containing:
            - data: Dict with keys:
                - df_train: Training DataFrame
                - df_valid: Validation DataFrame
                - target_column: str
                - model_name: str (one of: 'xgboost', 'lightgbm', 'random_forest', 'catboost')
                - custom_instructions: str or None
        :return: Dictionary with training results and metrics
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        required_keys = ['df_train', 'df_valid', 'target_column', 'model_name']
        if not all(key in task.data for key in required_keys):
            raise ValueError(f"Task data must contain: {required_keys}")

        # Get training code from LLM
        training_code, explanation = self._get_training_code(task.data)
        
        # Get target column from task data
        target_column = task.data['target_column']
        
        # Create namespace for execution
        globals_dict = {
            'train_df': task.data['df_train'].copy(),
            'valid_df': task.data['df_valid'].copy(),
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
        
        # Get results from namespace
        metrics = globals_dict.get('metrics', {
            'error': 'Metrics not created',
            'roc_auc': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'confusion_matrix': None
        })
        
        # Get the trained model object and training features
        model = globals_dict.get('model', None)
        training_features = globals_dict.get('training_features', [])
        
        if not model or not metrics:
            raise ValueError("Code execution failed to produce required outputs")
        
        return {
            'metrics': metrics,
            'model': model,
            'training_features': training_features  # Add training features to return dict
        }
            

    def _get_training_code(self, data: Dict) -> tuple[str, str]:
        """Get Python code for model training from LLM"""
        # Prepare data info for LLM
        data_info = self._get_data_info(data)
        
        # Prepare simplified arguments for prompt
        prompt_args = {
            'model_name': data['model_name'],
            'data_info': data_info,
            'target_column': data['target_column'],
            'date_column': data.get('date_column', 'billing_date'),  # Default or from mappings
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

        response = self._get_llm_response(system_prompt, user_prompt)
        print(">>>>>>response", response)
        return self._parse_llm_response(response)

    def _get_data_info(self, data: Dict) -> Dict:
        """Gather information about the datasets"""
        train_df = data['df_train']
        valid_df = data['df_valid']
        target_col = data['target_column']
        
        return {
            'train_shape': train_df.shape,
            'valid_shape': valid_df.shape,
            'target_distribution_train': train_df[target_col].value_counts().to_dict(),
            'target_distribution_valid': valid_df[target_col].value_counts().to_dict(),
            'features': [col for col in train_df.columns if col != target_col],
            'dtypes': train_df.dtypes.to_dict(),
            'column_mappings': {
                'target_column': target_col,
                'date_column': 'billing_date',
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