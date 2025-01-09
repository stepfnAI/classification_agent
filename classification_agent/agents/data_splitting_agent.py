from typing import Dict
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL,DEFAULT_LLM_PROVIDER
import json
import numpy as np
import re

class SFNDataSplittingAgent(SFNAgent):
    """Agent responsible for splitting data into train, validation, and inference sets"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER, **kwargs):
        super().__init__(name="Data Splitting", role="Data Splitter")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["data_splitter"]
        self.validation_window = kwargs.get('validation_window', 3)
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Generate and execute data splitting code"""
        if not isinstance(task.data, dict) or 'df' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' key")

        df = task.data['df']
        date_column = task.data.get('date_column')
        
        # Get data info for LLM
        data_info = {
            'total_records': len(df),
            'has_date': date_column is not None,
            'date_column': date_column,
            'validation_window': self.validation_window
        }
        
        # Get splitting code from LLM
        split_code, explanation = self._get_split_code(data_info)
        
        try:
            # Create local copy of dataframe for execution
            locals_dict = {
                'df': df.copy(),
                'np': np,
                'pd': pd,
                'date_column': date_column,
                'validation_window': self.validation_window
            }
            
            # Execute the code
            exec(split_code, globals(), locals_dict)
            
            # Verify the required DataFrames exist
            required_dfs = ['train_df', 'valid_df', 'infer_df']
            if not all(df_name in locals_dict for df_name in required_dfs):
                raise ValueError("Code execution did not produce all required DataFrames")
            
            # Get split information
            split_info = {
                'train_samples': len(locals_dict['train_df']),
                'valid_samples': len(locals_dict['valid_df']),
                'infer_samples': len(locals_dict['infer_df']),
                'train_start': locals_dict['train_df'][date_column].min() if date_column else None,
                'train_end': locals_dict['train_df'][date_column].max() if date_column else None,
                'valid_start': locals_dict['valid_df'][date_column].min() if date_column else None,
                'valid_end': locals_dict['valid_df'][date_column].max() if date_column else None,
                'infer_month': locals_dict['infer_df'][date_column].dt.to_period('M').iloc[0] if date_column else None,
                # Add DataFrames to split info
                'train_df': locals_dict['train_df'],
                'valid_df': locals_dict['valid_df'],
                'infer_df': locals_dict['infer_df']
            }
            
            return split_info
            
        except Exception as e:
            print(f"Error in code execution: {str(e)}")
            print(f"Generated code:\n{split_code}")
            raise ValueError(f"Error executing splitting code: {str(e)}")

    def _get_data_info(self, df: pd.DataFrame, field_mappings: Dict, target_column: str) -> Dict:
        """Gather information about the dataset for LLM"""
        date_column = field_mappings.get('date')
        info = {
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'date_column': date_column,
            'has_date': bool(date_column and date_column in df.columns),
            'target_column': target_column,
            'missing_target_count': df[target_column].isna().sum() if target_column else 0
        }
        
        if info['has_date']:
            try:
                dates = pd.to_datetime(df[date_column])
                info.update({
                    'date_range': f"{dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}",
                    'unique_months': len(dates.dt.to_period('M').unique())
                })
            except:
                info['has_date'] = False
                
        return info

    def _get_split_code(self, data_info: Dict, manual_instructions: str = None) -> tuple[str, str]:
        """Get Python code for splitting from LLM"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='data_splitter',
            llm_provider=self.llm_provider,
            prompt_type='main',
            data_info=data_info,
            validation_window=self.validation_window,
            manual_instructions=manual_instructions or "None provided"
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
        print(">>response_for_debug", response)
        return self._parse_llm_response(response)

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
            print(">>uncleaned_code_for_debug_code", code)
            # Clean the code
            code = self._clean_code(code)
            print(">>cleaned_code_for_debug_code", code)
            return code, explanation
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            # Return default random split code
            return self._get_default_split_code()

    def _clean_code(self, code: str) -> str:
        """Clean the generated code"""
        # Remove markdown code blocks
        code = re.sub(r'```python\n|```', '', code)
        
        # Remove comments and print statements
        code = re.sub(r'print\(.*\)\n?', '', code)
        code = re.sub(r'#.*\n', '', code)
        
        # Split into lines and remove empty lines
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Fix indentation
        cleaned_lines = []
        in_try_block = False
        for line in lines:
            if line.strip().startswith('try:'):
                in_try_block = True
                continue
            elif line.strip().startswith('except'):
                in_try_block = False
                continue
            
            # If we're in a try block, remove one level of indentation
            if in_try_block:
                # Remove one level of indentation (typically 4 spaces or 1 tab)
                line = re.sub(r'^    ', '', line)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _get_default_split_code(self) -> tuple[str, str]:
        """Return default random split code"""
        code = """
        indices = np.random.permutation(len(df))
        train_size = int(len(df) * 0.7)
        valid_size = int(len(df) * 0.2)
        train_df = df.iloc[indices[:train_size]]
        valid_df = df.iloc[indices[train_size:train_size + valid_size]]
        infer_df = df.iloc[indices[train_size + valid_size:]]
        """
        return code.strip(), "Performed random split (70-20-10) due to error in custom split" 

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        if not isinstance(task.data, dict) or 'df' not in task.data:
            raise ValueError("Task data must contain df")

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='data_splitter',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 