from typing import Dict, List
import streamlit as st
from sfn_blueprint import Task
from classification_agent.agents.model_training_agent import SFNModelTrainingAgent
from classification_agent.utils.model_manager import ModelManager

class ModelTrainingOrchestrator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.model_training_agent = SFNModelTrainingAgent()

    def execute(self) -> Dict:
        """Execute the model training step"""
        st.header("Step 5: Model Training")
        
        if not self.model_manager.has_split_data():
            st.error("Please complete data splitting first!")
            return {}

        # Get split data
        train_df = self.model_manager.get_train_data()
        valid_df = self.model_manager.get_validation_data()
        target_column = self.model_manager.get_target_column()

        if train_df is None or valid_df is None:
            st.error("Training or validation data not found!")
            return {}

        # List of models to train
        models_to_train = ['xgboost', 'lightgbm', 'random_forest', 'catboost']
        results = {}

        with st.spinner("Training models..."):
            for model_name in models_to_train:
                st.write(f"\nTraining {model_name.upper()} model...")
                
                # Prepare task data
                task_data = {
                    'df_train': train_df,
                    'df_valid': valid_df,
                    'target_column': target_column,
                    'model_name': model_name
                }

                # Create and execute task
                task = Task(data=task_data)
                try:
                    result = self.model_training_agent.execute_task(task)
                    
                    # Store results
                    results[model_name] = result
                    
                    # Display metrics
                    self._display_model_metrics(model_name, result['metrics'])
                    
                    # Store model if valid
                    if result['model'] is not None:
                        self.model_manager.add_trained_model(
                            model_name=model_name,
                            model=result['model'],
                            metrics=result['metrics'],
                            training_features=result['training_features']
                        )
                except Exception as e:
                    st.error(f"Error training {model_name} model: {str(e)}")
                    continue

        if results:
            st.success("Model training completed!")
            self.model_manager.save_state()
        else:
            st.error("No models were successfully trained!")

        return results

    def _display_model_metrics(self, model_name: str, metrics: Dict):
        """Display model metrics in a formatted way"""
        try:
            st.write(f"\n{model_name.upper()} Model Metrics:")
            
            if not isinstance(metrics, dict):
                st.error(f"Invalid metrics format for {model_name}")
                return
            
            # Helper function to format metric value
            def format_metric(value):
                if value is None or value == 'N/A' or value == '':
                    return 'N/A'
                try:
                    float_val = float(value)
                    return f"{float_val:.3f}" if float_val != float('inf') else 'N/A'
                except (ValueError, TypeError):
                    return str(value)
            
            # Display metrics safely
            metrics_to_display = {
                'AUC': metrics.get('roc_auc'),
                'Precision': metrics.get('precision'),
                'Recall': metrics.get('recall'),
                'F1 Score': metrics.get('f1')
            }
            
            for metric_name, value in metrics_to_display.items():
                formatted_value = format_metric(value)
                st.write(f"- {metric_name}: {formatted_value}")
            
            # Handle confusion matrix separately
            conf_matrix = metrics.get('confusion_matrix')
            if conf_matrix is not None:
                st.write("\nConfusion Matrix:")
                try:
                    st.write(str(conf_matrix))
                except:
                    st.write("Unable to display confusion matrix")
            
            # Display any error messages if present
            error_msg = metrics.get('error')
            if error_msg and error_msg != 'Metrics not created':
                st.error(f"Error: {error_msg}")
            
        except Exception as e:
            st.error(f"Error displaying metrics for {model_name}: {str(e)}") 