from typing import Dict
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json

class SFNModelSelectionAgent(SFNAgent):
    """Agent responsible for selecting the best model based on performance metrics"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Model Selection", role="ML Advisor")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["model_selector"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """
        Select best model based on performance metrics and requirements
        
        :param task: Task object containing:
            - data: Dict with keys:
                - model_results: Dict[str, Dict] - Results from each model
                    {
                        'model_name': {
                            'metrics': {
                                'auc': float,
                                'precision': float,
                                'recall': float,
                                'f1': float,
                                'confusion_matrix': List[List[int]]
                            },
                            'training_time': str,
                            'model_params': dict
                        }
                    }
                - target_distribution: Dict - Class distribution
                - custom_instructions: str or None - Any specific selection criteria
        :return: Dictionary with selection results and explanation
        """
        if not isinstance(task.data, dict) or 'model_results' not in task.data:
            raise ValueError("Task data must be a dictionary containing model_results")

        # Get selection recommendation from LLM
        selection_info = self._get_model_selection(task.data)
        
        return {
            'best_model': selection_info['selected_model'],
            'explanation': selection_info['explanation'],
            'comparison_summary': selection_info['comparison_summary'],
            'model_rankings': selection_info['model_rankings']
        }

    def _get_model_selection(self, data: Dict) -> Dict:
        """Get model selection recommendation from LLM"""
        # Prepare info for LLM
        selection_info = self._prepare_selection_info(data)
        
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='model_selector',
            llm_provider=self.llm_provider,
            prompt_type='main',
            selection_info=selection_info,
            custom_instructions=data.get('custom_instructions', 'None provided')
        )

        response = self._get_llm_response(system_prompt, user_prompt)
        return self._parse_llm_response(response)

    def _prepare_selection_info(self, data: Dict) -> Dict:
        """Prepare model comparison information"""
        model_results = data['model_results']
        target_distribution = data.get('target_distribution', {})
        
        return {
            'models': {
                model_name: {
                    'metrics': results['metrics'],
                    'training_time': results['training_time'],
                    'parameters': results['model_params']
                }
                for model_name, results in model_results.items()
            },
            'target_distribution': target_distribution,
            'total_models': len(model_results)
        }

    def _get_llm_response(self, system_prompt: str, user_prompt: str):
        """Get response from LLM"""
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
        
        return response

    def _parse_llm_response(self, response) -> Dict:
        """Parse LLM response into selection results"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response
                
            return json.loads(content)
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            raise ValueError("Failed to parse LLM response")

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        if not isinstance(task.data, dict) or 'model_results' not in task.data:
            raise ValueError("Task data must contain model_results")

        prompts = self.prompt_manager.get_prompt(
            agent_type='model_selector',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 