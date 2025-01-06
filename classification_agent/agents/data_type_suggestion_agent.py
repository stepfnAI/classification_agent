from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json

class SFNDataTypeSuggestionAgent(SFNAgent):
    """Agent responsible for suggesting data type conversions and modifications"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Data Type Suggestion", role="Data Type Analyzer")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["data_type_suggester"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict[str, List[str]]:
        """
        Analyzes DataFrame and suggests data type conversions
        Returns: Dictionary with conversion instructions
        """
        if not isinstance(task.data, pd.DataFrame):
            raise ValueError("Task data must be a pandas DataFrame")

        df = task.data
        data_info = self._get_data_info(df)
        conversion_instructions = self._generate_conversion_instructions(data_info)
        return conversion_instructions

    def _get_data_info(self, df: pd.DataFrame) -> Dict:
        """Gather information about DataFrame columns and their current data types"""
        info = {
            'dtypes': df.dtypes.to_dict(),
            'sample_values': {col: df[col].head(3).tolist() for col in df.columns},
            'unique_counts': {col: df[col].nunique() for col in df.columns},
            'null_counts': df.isnull().sum().to_dict()
        }
        return info

    def _generate_conversion_instructions(self, data_info: Dict) -> Dict[str, List[str]]:
        """Generate data type conversion instructions based on data analysis"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='data_type_suggester',
            llm_provider=self.llm_provider,
            prompt_type='main',
            data_info=data_info
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
                "conversions": [],
                "validations": []
            }

    def _validate_instructions(self, instructions: Dict) -> Dict[str, List[str]]:
        """Validate and normalize the conversion instructions"""
        validated = {
            "conversions": [],  # List of conversion steps
            "validations": []   # List of validation checks
        }
        
        if isinstance(instructions.get('conversions'), list):
            validated['conversions'] = instructions['conversions']
        if isinstance(instructions.get('validations'), list):
            validated['validations'] = instructions['validations']
            
        return validated

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        if not isinstance(task.data, pd.DataFrame):
            raise ValueError("Task data must be a pandas DataFrame")

        prompts = self.prompt_manager.get_prompt(
            agent_type='data_type_suggester',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 