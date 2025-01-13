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
from sklearn.model_selection import train_test_split

class SFNDataSplittingAgent(SFNAgent):
    """Agent responsible for splitting data into train, validation, and inference sets"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER, **kwargs):
        super().__init__(name="Data Splitting", role="Data Splitter")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["data_splitter"]
        self.validation_window = kwargs.get('validation_window', 3)
        self.max_retries = kwargs.get('max_retries', 3)
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Generate and execute data splitting code"""
        if not isinstance(task.data, dict) or 'df' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' key")

        df = task.data['df']
        field_mappings = task.data.get('field_mappings', {})
        target_column = field_mappings.get('target')
        date_column = field_mappings.get('date')
        user_instructions = task.data.get('user_instructions', '')  # Get user instructions
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if last_error:
                    print(f">>>>>> Retry attempt {attempt + 1}/{self.max_retries}. Previous error: {str(last_error)}")

                split_code, explanation = self._get_split_code(
                    total_records=str(len(df)),
                    columns=', '.join(df.columns.tolist()),
                    field_mappings=str(field_mappings),
                    target_column=str(target_column),
                    date_column=str(date_column),
                    validation_window=self.validation_window,
                    user_instructions=user_instructions,
                    previous_error=str(last_error) if last_error else None,
                    attempt=attempt + 1
                )

                # Create local copy of dataframe for execution with all necessary imports
                locals_dict = {
                    'df': df.copy(),
                    'np': np,
                    'pd': pd,
                    'train_test_split': train_test_split,
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
                    'infer_df': locals_dict['infer_df'],
                    # Add explanation to split info
                    'explanation': explanation
                }
                
                print(f">>>>>> Successfully split data on attempt {attempt + 1}")
                return split_info

            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    # If all retries failed, use default split
                    default_code, default_explanation = self._get_default_split_code()
                    try:
                        exec(default_code, globals(), locals_dict)
                        print(">>>>>> Successfully split data using default split after all retries failed")
                        return {
                            'train_df': locals_dict['train_df'],
                            'valid_df': locals_dict['valid_df'],
                            'infer_df': locals_dict['infer_df'],
                            'explanation': default_explanation
                        }
                    except Exception as default_error:
                        raise ValueError(f"Both custom and default splits failed. Last error: {str(default_error)}")

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

    def _get_split_code(self, total_records, columns, field_mappings, target_column, 
                       date_column, validation_window, user_instructions, 
                       previous_error=None, attempt=1) -> tuple[str, str]:
        """Get Python code for splitting from LLM"""
        print(f"1>>>>>Total records: {total_records}")
        print(f"1>>>>>Columns: {columns}")
        print(f"1>>>>>Field mappings: {field_mappings}")
        print(f"1>>>>>Target column: {target_column}")
        print(f"1>>>>>Date column: {date_column}")
        print(f"1>>>>>Validation window: {validation_window}")
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='data_splitter',
            llm_provider=self.llm_provider,
            prompt_type='main',
            total_records=total_records,
            columns=columns,
            field_mappings=field_mappings,
            target_column=target_column,
            date_column=date_column,
            validation_window=validation_window,
            user_instructions=user_instructions
        )

        if previous_error and attempt > 1:
            user_prompt += f"\n\nPrevious attempt {attempt-1} failed with error:\n{previous_error}\nPlease adjust the code to handle this error."

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
        # code = re.sub(r'#.*\n', '', code)
        
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