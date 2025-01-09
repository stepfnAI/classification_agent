from typing import Dict
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
from classification_agent.utils.model_manager import ModelManager
import json

class SFNModelSelectionAgent(SFNAgent):
    """Agent responsible for analyzing model performance and making recommendations"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Model Selection", role="ML Strategist")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["model_selector"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        self.model_manager = ModelManager()

    def execute_task(self, task: Task) -> Dict:
        """
        Analyze model results and recommend the best model
        
        :param task: Task object containing:
            - data: Dict with model_results from training step
        :return: Dictionary with recommendation details
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        model_results = task.data.get('model_results', {})
        if not model_results:
            raise ValueError("No model results found for analysis")

        # Format model results for LLM
        selection_info = self._format_model_results(model_results)
        
        # Get recommendation from LLM
        recommendation = self._get_recommendation(selection_info)
        
        return recommendation

    def _format_model_results(self, model_results: Dict) -> str:
        """Format model results for LLM consumption"""
        formatted_info = "Model Performance Summary:\n\n"
        
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            formatted_info += f"Model: {model_name}\n"
            formatted_info += f"- ROC AUC: {metrics.get('roc_auc', 'N/A')}\n"
            formatted_info += f"- F1 Score: {metrics.get('f1', 'N/A')}\n"
            formatted_info += f"- Precision: {metrics.get('precision', 'N/A')}\n"
            formatted_info += f"- Recall: {metrics.get('recall', 'N/A')}\n\n"
        
        return formatted_info

    def _get_recommendation(self, selection_info: str) -> Dict:
        """Get model recommendation from LLM"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='model_selector',
            llm_provider=self.llm_provider,
            prompt_type='main',
            selection_info=selection_info,
            custom_instructions=""
        )

        response = self._get_llm_response(system_prompt, user_prompt)
        return self._parse_llm_response(response)

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
        """Parse LLM response into structured recommendation"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
            
            # Clean response by removing markdown code blocks and 'json' word
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
            
            # Parse the cleaned JSON content
            recommendation = json.loads(content)  # Using json.loads instead of eval for safety
            return recommendation
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw content:\n{content}")
            raise ValueError("Failed to parse LLM response") 

    def get_validation_params(self, response, task):
        """Get parameters for validation
        
        :param response: The recommendation response to validate
        :param task: Task object containing the model results
        :return: Dictionary with validation parameters
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='model_selector',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 