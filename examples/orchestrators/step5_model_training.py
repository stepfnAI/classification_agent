from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.model_training_agent import SFNModelTrainingAgent
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER
from classification_agent.utils.model_manager import ModelManager
import pandas as pd

class ModelTraining:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.training_agent = SFNModelTrainingAgent()
        self.model_pool = ['xgboost', 'lightgbm', 'random_forest', 'catboost']
        self.model_manager = ModelManager()
        
    def execute(self):
        """Orchestrates the model training step"""
        # Get split info and data
        split_info = self.session.get('split_info')
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        print(f"Available mappings for model training: {mappings}")
        
        if not split_info:
            self.view.show_message("❌ Data split information not found.", "error")
            return False
            
        # Train models if not done
        if not self.session.get('models_trained'):
            all_model_results = {}
            
            for model_name in self.model_pool:
                with self.view.display_spinner(f'🤖 Training {model_name}...'):
                    validate_and_retry_agent = SFNValidateAndRetryAgent(
                        llm_provider=DEFAULT_LLM_PROVIDER,
                        for_agent='model_trainer'
                    )
                    
                    task = Task("Train models", data={
                        'df_train': split_info['train_df'],
                        'df_valid': split_info['valid_df'],
                        'target_column': mappings.get('target'),
                        'model_name': model_name
                    })
                    
                    result, validation_message, is_valid = validate_and_retry_agent.complete(
                        agent_to_validate=self.training_agent,
                        task=task,
                        validation_task=task,
                        method_name='execute_task',
                        get_validation_params='get_validation_params',
                        max_retries=2,
                        retry_delay=3.0
                    )
                    
                    if not is_valid:
                        self.view.show_message(f"❌ Training failed for {model_name}: {validation_message}", "error")
                        continue
                    
                    # Save model with metadata including training features
                    model_id = self.model_manager.save_model(
                        model=result['model'],
                        model_name=model_name,
                        metadata={
                            'metrics': result['metrics'],
                            'features': result['training_features'],
                            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
                        }
                    )
                    
                    # Store results
                    all_model_results[model_name] = {
                        'model_id': model_id,
                        'metrics': result['metrics']
                    }
                    
                    # Display metrics
                    self._display_model_metrics(model_name, result['metrics'])
            
            # Store all results in session
            self.session.set('model_results', all_model_results)
            self.session.set('models_trained', True)
            
        # If complete, mark step as done
        if not self.session.get('step_5_complete'):
            self._save_step_summary()
            self.session.set('step_5_complete', True)
            
        return True
        
    def _display_model_metrics(self, model_name: str, metrics: dict):
        """Display metrics for a trained model"""
        self.view.display_subheader(f"{model_name} Results")
        
        metrics_text = f"**Metrics:**\n"
        metrics_text += f"- AUC: {metrics.get('roc_auc', 'N/A'):.3f}\n"
        metrics_text += f"- F1 Score: {metrics.get('f1', 'N/A'):.3f}\n"
        metrics_text += f"- Precision: {metrics.get('precision', 'N/A'):.3f}\n"
        metrics_text += f"- Recall: {metrics.get('recall', 'N/A'):.3f}\n"
        
        self.view.show_message(metrics_text, "info")
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        model_results = self.session.get('model_results', {})
        
        summary = "✅ Model Training Complete:\n"
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            summary += f"- {model_name}:\n"
            summary += f"  - AUC: **{metrics.get('roc_auc', 'N/A')}**\n"
            summary += f"  - F1: **{metrics.get('f1', 'N/A')}**\n"
            summary += f"  - Precision: **{metrics.get('precision', 'N/A')}**\n"
            summary += f"  - Recall: **{metrics.get('recall', 'N/A')}**\n"
        
        self.session.set('step_5_summary', summary) 