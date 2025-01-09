from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.categorical_feature_agent import SFNCategoricalFeatureAgent
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER

class FeaturePreprocessing:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.categorical_agent = SFNCategoricalFeatureAgent()
        
    def execute(self):
        """Orchestrates the feature preprocessing step"""
        # Handle categorical features if not done
        if not self.session.get('categorical_features_complete'):
            if not self._handle_categorical_features():
                return False
        
        # If complete, mark step as done
        if not self.session.get('step_3_complete'):
            self._save_step_summary()
            self.session.set('step_3_complete', True)
            
        return True
        
    def _handle_categorical_features(self):
        """Handle categorical feature processing"""
        try:
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            
            with self.view.display_spinner('🤖 AI is analyzing categorical features...'):
                # Create task for categorical feature analysis
                task_data = {
                    'df': df,
                    'mappings': mappings
                }
                
                result = self.categorical_agent.execute_task(Task("Analyze categorical features", data=task_data))
                
                modified_df = result['df']
                feature_info = result['feature_info']
            
            # Display modified data and feature information
            self.view.display_subheader("Categorical Feature Processing")
            self.view.display_dataframe(modified_df.head())
            
            # Show encoding summary
            self.view.display_subheader("Encoding Summary")
            summary_msg = "Applied Encodings:\n"
            for feature, info in feature_info.items():
                summary_msg += f"- {feature}: **{info['encoding_type']}**\n"
                if 'cardinality' in info:
                    summary_msg += f"  - Unique values: {info['cardinality']}\n"
            self.view.show_message(summary_msg, "info")
            
            if self.view.display_button("Confirm Feature Processing"):
                self.session.set('df', modified_df)
                self.session.set('feature_info', feature_info)
                self.session.set('categorical_features_complete', True)
                return True
                
        except Exception as e:
            self.view.show_message(f"Error in categorical feature processing: {str(e)}", "error")
            return False
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        feature_info = self.session.get('feature_info', {})
        
        summary = "✅ Feature Preprocessing Complete:\n"
        for feature, info in feature_info.items():
            summary += f"- {feature}: **{info['encoding_type']}**\n"
        self.session.set('step_3_summary', summary) 