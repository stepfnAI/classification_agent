import os
import pandas as pd
from sfn_blueprint import SFNAgent
import re
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG
from sfn_blueprint.utils.openai_client import SFNOpenAIClient

class SFNCodeGeneratorAgent(SFNAgent):
    """Agent responsible for generating executable Python code from instructions"""
    
    def __init__(self):
        super().__init__(name="Code Generator", role="Python Developer")
        self.client = SFNOpenAIClient()
        self.model_config = MODEL_CONFIG["code_generator"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task, llm_provider='openai', error_message=None) -> str:
        """
        Generate executable Python code from instructions
        
        :param task: Task object containing:
            - data: Dict with keys:
                - instructions: List of conversion instructions
                - df_info: Dict with DataFrame information
                - error_message: Optional error message from previous attempt
        :return: Generated Python code as string
        """
        # Prepare kwargs for prompt
        prompt_kwargs = {
            'instructions': task.data['instructions'],
            'df_info': task.data['df_info'],
            'error_message': error_message
        }
        
        # Get prompts using SFNPromptManager
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='code_generator',
            llm_provider=llm_provider,
            prompt_type='main',
            **prompt_kwargs
        )
        
        response = self.client.chat.completions.create(
            model=self.model_config[llm_provider]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.model_config[llm_provider]["temperature"],
            max_tokens=self.model_config[llm_provider]["max_tokens"]
        )

        code = response.choices[0].message.content.strip()
        return self.clean_generated_code(code)
    
    @staticmethod
    def clean_generated_code(code: str) -> str:
        """Clean and format the generated code"""
        # Remove markdown code blocks
        code = re.sub(r'```python\n|```', '', code)
        # Remove print statements
        code = re.sub(r'print\(.*\)\n?', '', code)
        # Remove comments
        code = re.sub(r'#.*\n', '', code)
        # Remove empty lines
        code = '\n'.join(line for line in code.split('\n') if line.strip())
        return code 

    def get_validation_params(self, response, task):
        """
        Get parameters for validation
        
        :param response: The generated code to validate
        :param task: Task object containing the original instructions and DataFrame info
        :return: Dictionary with validation parameters
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='code_generator',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            instructions=task.data.get('instructions', []),
            df_info=task.data.get('df_info', {})
        )
        return prompts 