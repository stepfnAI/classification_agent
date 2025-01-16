from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from classification_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
import json

class SFNLeakageDetectionAgent(SFNAgent):
    """Agent responsible for detecting potential target leakage in features"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Leakage Detection", role="Data Validator")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["leakage_detector"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        self.CORRELATION_THRESHOLD = 0.95
        
    def execute_task(self, task: Task) -> Dict:
        """Analyze features for potential target leakage"""
        if not isinstance(task.data.get('df'), pd.DataFrame):
            raise ValueError("Task data must contain a pandas DataFrame under 'df' key")

        df = task.data['df']
        target_col = task.data['target_column']
        
        # Get comprehensive statistical analysis
        statistical_findings = self._get_statistical_findings(df, target_col)
        
        # Get LLM analysis
        leakage_analysis = self._analyze_leakage(
            statistical_findings=statistical_findings,
            field_mappings=task.data['field_mappings']
        )
        
        return leakage_analysis

    def _get_statistical_findings(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Calculate comprehensive statistical metrics for leakage detection"""
        findings = {
            'correlations': {},
            'unique_value_counts': {},
            'perfect_predictors': [],
            'near_perfect_predictors': [],
            'high_cardinality_features': [],
            'target_statistics': {},
            'temporal_indicators': {}
        }
        
        # Basic target statistics
        findings['target_statistics'] = {
            'unique_values': df[target_col].nunique(),
            'null_count': df[target_col].isnull().sum(),
            'value_distribution': df[target_col].value_counts(normalize=True).to_dict()
        }

        for column in df.columns:
            if column == target_col:
                continue
                
            # 1. Calculate correlations based on data type
            correlation = self._calculate_correlation(df[column], df[target_col])
            findings['correlations'][column] = correlation
            
            # 2. Get unique value counts and cardinality metrics
            unique_count = df[column].nunique()
            null_count = df[column].isnull().sum()
            findings['unique_value_counts'][column] = {
                'unique_count': unique_count,
                'null_count': null_count,
                'cardinality_ratio': unique_count / len(df) if len(df) > 0 else 0
            }
            
            # 3. Check for perfect/near-perfect predictors
            if abs(correlation) > self.CORRELATION_THRESHOLD:
                if abs(correlation) > 0.99:
                    findings['perfect_predictors'].append(column)
                else:
                    findings['near_perfect_predictors'].append(column)
            
            # 4. Identify high cardinality features
            if findings['unique_value_counts'][column]['cardinality_ratio'] > 0.9:
                findings['high_cardinality_features'].append(column)
            
            # 5. Check for potential temporal indicators
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                findings['temporal_indicators'][column] = {
                    'is_monotonic': df[column].is_monotonic,
                    'missing_timestamps': null_count,
                    'date_range': {
                        'start': df[column].min().strftime('%Y-%m-%d') if not pd.isna(df[column].min()) else None,
                        'end': df[column].max().strftime('%Y-%m-%d') if not pd.isna(df[column].max()) else None
                    }
                }
            
            # 6. Additional statistical measures
            if pd.api.types.is_numeric_dtype(df[column]):
                findings['correlations'][f"{column}_stats"] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'skewness': df[column].skew(),
                    'unique_ratio': unique_count / len(df)
                }

        return findings

    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate correlation between two series handling different data types
        Returns correlation coefficient between -1 and 1
        """
        try:
            # For numeric data, use standard correlation
            if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
                return series1.corr(series2)
            
            # For categorical data, use Cramer's V
            elif pd.api.types.is_categorical_dtype(series1) or pd.api.types.is_categorical_dtype(series2):
                contingency = pd.crosstab(series1, series2)
                chi2 = stats.chi2_contingency(contingency)[0]
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                return cramer_v
            
            # For datetime, convert to ordinal and calculate correlation
            elif pd.api.types.is_datetime64_any_dtype(series1):
                return series1.map(pd.Timestamp.toordinal).corr(series2)
            
            # For other types, try to convert to numeric
            else:
                try:
                    return pd.to_numeric(series1, errors='coerce').corr(
                        pd.to_numeric(series2, errors='coerce')
                    )
                except:
                    return 0.0
                    
        except Exception as e:
            print(f"Error calculating correlation: {str(e)}")
            return 0.0

    def _analyze_leakage(self, statistical_findings: Dict, field_mappings: Dict) -> Dict:
        """Get LLM analysis of potential leakage"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='leakage_detector',
            llm_provider=self.llm_provider,
            prompt_type='main',
            statistical_findings=statistical_findings,
            field_mappings=field_mappings
        )

        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 500,
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
            # Extract content from response based on response type
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response
            
            # Parse LLM analysis
            llm_analysis = json.loads(content)
            
            # Combine statistical findings with LLM insights
            return {
                'flagged_features': statistical_findings['perfect_predictors'] + 
                                  statistical_findings['near_perfect_predictors'],
                'explanations': {
                    'perfect_predictors': statistical_findings['perfect_predictors'],
                    'near_perfect': statistical_findings['near_perfect_predictors'],
                    'high_cardinality': statistical_findings['high_cardinality_features'],
                    'temporal_indicators': statistical_findings['temporal_indicators'],
                    'llm_insights': llm_analysis.get('insights', {}),  # LLM's additional insights
                    'reasoning': llm_analysis.get('reasoning', {})     # LLM's reasoning
                },
                'recommendations': {
                    'remove': statistical_findings['perfect_predictors'],
                    'review': statistical_findings['near_perfect_predictors'] + 
                             llm_analysis.get('additional_review', []),  # LLM's suggestions
                    'monitor': statistical_findings['high_cardinality_features'],
                    'llm_suggestions': llm_analysis.get('recommendations', {})  # LLM's recommendations
                }
            }
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            # Fallback to statistical findings only
            return {
                'flagged_features': statistical_findings['perfect_predictors'] + 
                                  statistical_findings['near_perfect_predictors'],
                'explanations': {
                    'perfect_predictors': statistical_findings['perfect_predictors'],
                    'near_perfect': statistical_findings['near_perfect_predictors'],
                    'high_cardinality': statistical_findings['high_cardinality_features'],
                    'temporal_indicators': statistical_findings['temporal_indicators']
                },
                'recommendations': {
                    'remove': statistical_findings['perfect_predictors'],
                    'review': statistical_findings['near_perfect_predictors'],
                    'monitor': statistical_findings['high_cardinality_features']
                }
            }

    def get_validation_params(self, response, task):
        """
        Get parameters for validation
        :param response: The response from execute_task to validate
        :param task: The validation task containing the analysis data
        :return: Dictionary with validation parameters
        """
        if not isinstance(task.data.get('df'), pd.DataFrame):
            raise ValueError("Task data must contain a pandas DataFrame under 'df' key")

        prompts = self.prompt_manager.get_prompt(
            agent_type='leakage_detector',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            statistical_findings=self._get_statistical_findings(
                task.data['df'], 
                task.data['target_column']
            )
        )
        return prompts 