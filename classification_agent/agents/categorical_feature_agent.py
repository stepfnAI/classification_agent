from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json

class SFNCategoricalFeatureAgent(SFNAgent):
    """Agent responsible for suggesting categorical feature processing strategies"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Categorical Feature Handler", role="Feature Engineer")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["categorical_feature_handler"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict[str, List[str]]:
        """
        Analyzes DataFrame and suggests categorical feature processing strategies
        
        :param task: Task object containing:
            - data: Dict with keys:
                - df: pandas DataFrame
                - field_mappings: Dict of field mappings (optional)
                - ignore_columns: List of columns to ignore (optional)
        :return: Dictionary with encoding instructions
        """
        if not isinstance(task.data, dict) or 'df' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' key with a pandas DataFrame")

        df = task.data['df']
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        # Get optional parameters
        field_mappings = task.data.get('field_mappings', {})
        ignore_columns = task.data.get('ignore_columns', [])
        
        # Add mapped fields to ignore list
        ignore_columns.extend(field_mappings.values())
        
        categorical_info = self._get_categorical_info(df, ignore_columns)
        encoding_instructions = self._generate_encoding_instructions(categorical_info)
        return encoding_instructions

    def _get_categorical_info(self, df: pd.DataFrame, ignore_columns: List[str]) -> Dict:
        """Gather information about categorical columns"""
        # Filter out ignored columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns if col not in ignore_columns]
        
        info = {
            'categorical_columns': categorical_columns,
            'cardinality': {col: df[col].nunique() for col in categorical_columns},
            'sample_values': {col: df[col].value_counts().head(5).to_dict() for col in categorical_columns},
            'null_counts': df[categorical_columns].isnull().sum().to_dict()
        }
        return info

    def _generate_encoding_instructions(self, cat_info: Dict) -> Dict[str, List[str]]:
        """Generate categorical encoding instructions based on analysis"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='categorical_feature_handler',
            llm_provider=self.llm_provider,
            prompt_type='main',
            categorical_info=cat_info
        )

        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
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
        
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response
                
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
            
            instructions = json.loads(cleaned_str)
            return self._validate_instructions(instructions)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {
                "encodings": [],
                "validations": []
            }

    def _validate_instructions(self, instructions: Dict) -> Dict[str, List[str]]:
        """Validate and normalize the encoding instructions"""
        validated = {
            "encodings": [],  # List of encoding steps
            "validations": []  # List of validation checks
        }
        
        if isinstance(instructions.get('encodings'), list):
            validated['encodings'] = instructions['encodings']
        if isinstance(instructions.get('validations'), list):
            validated['validations'] = instructions['validations']
            
        return validated

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        if not isinstance(task.data, dict) or 'df' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' key")

        prompts = self.prompt_manager.get_prompt(
            agent_type='categorical_feature_handler',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            df_info={
                'columns': task.data['df'].columns.tolist(),
                'field_mappings': task.data.get('field_mappings', {})
            }
        )
        return prompts 