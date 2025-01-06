from typing import Dict
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json
import re

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
        Train a specific model and return metrics
        
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
        
        try:
            # Create local namespace for execution
            locals_dict = {
                'train_df': task.data['df_train'].copy(),
                'valid_df': task.data['df_valid'].copy(),
                'pd': pd,
                'np': np
            }
            
            # Execute the training code
            exec(training_code, globals(), locals_dict)
            
            # Get results from local namespace
            return {
                'metrics': locals_dict['metrics'],
                'training_time': locals_dict['training_time'],
                'model_params': locals_dict['model_params'],
                'message': explanation
            }
            
        except Exception as e:
            raise ValueError(f"Error executing training code: {str(e)}")

    def _get_training_code(self, data: Dict) -> tuple[str, str]:
        """Get Python code for model training from LLM"""
        # Prepare data info for LLM
        data_info = self._get_data_info(data)
        
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='model_trainer',
            llm_provider=self.llm_provider,
            prompt_type='main',
            data_info=data_info,
            model_name=data['model_name'],
            custom_instructions=data.get('custom_instructions', 'None provided')
        )

        response = self._get_llm_response(system_prompt, user_prompt)
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
            'dtypes': train_df.dtypes.to_dict()
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
                
            # Extract code and explanation from response
            response_dict = json.loads(content)
            code = response_dict['code']
            explanation = response_dict['explanation']
            
            # Clean the code
            code = self._clean_code(code)
            
            return code, explanation
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            raise ValueError("Failed to parse LLM response")

    def _clean_code(self, code: str) -> str:
        """Clean the generated code"""
        code = re.sub(r'```python\n|```', '', code)
        code = re.sub(r'print\(.*\)\n?', '', code)
        code = re.sub(r'#.*\n', '', code)
        code = '\n'.join(line for line in code.split('\n') if line.strip())
        return code 