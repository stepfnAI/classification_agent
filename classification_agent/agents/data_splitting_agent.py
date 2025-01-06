from typing import Dict
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json
import numpy as np
import re

class SFNDataSplittingAgent(SFNAgent):
    """Agent responsible for splitting data into train, validation, and inference sets"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Data Splitting", role="Data Splitter")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["data_splitter"]
        self.validation_window = 3
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Generate and execute data splitting code"""
        if not isinstance(task.data, dict) or 'df' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' key")

        df = task.data['df']
        field_mappings = task.data.get('field_mappings', {})
        target_column = task.data.get('target_column')
        manual_instructions = task.data.get('manual_instructions')

        # Get data info for LLM
        data_info = self._get_data_info(df, field_mappings, target_column)
        
        # Get splitting code from LLM
        split_code, explanation = self._get_split_code(data_info, manual_instructions)
        
        # Execute the splitting code
        try:
            # Create local copy of dataframe for execution
            locals_dict = {'df': df.copy(), 'np': np, 'pd': pd}
            
            # Execute the code
            exec(split_code, globals(), locals_dict)
            
            # Get the split datasets
            splits = {
                'train': locals_dict['train_df'],
                'validation': locals_dict['valid_df'],
                'infer': locals_dict['infer_df'],
                'message': explanation
            }
            return splits
            
        except Exception as e:
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
            
            # Clean the code
            code = self._clean_code(code)
            
            return code, explanation
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            # Return default random split code
            return self._get_default_split_code()

    def _clean_code(self, code: str) -> str:
        """Clean the generated code"""
        code = re.sub(r'```python\n|```', '', code)
        code = re.sub(r'print\(.*\)\n?', '', code)
        code = re.sub(r'#.*\n', '', code)
        code = '\n'.join(line for line in code.split('\n') if line.strip())
        return code

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