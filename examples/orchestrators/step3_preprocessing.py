from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.categorical_feature_agent import SFNCategoricalFeatureAgent
from classification_agent.agents.leakage_detection_agent import SFNLeakageDetectionAgent
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER

class FeaturePreprocessing:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.categorical_agent = SFNCategoricalFeatureAgent()
        self.leakage_detector = SFNLeakageDetectionAgent()
        
    def execute(self):
        """Orchestrates the feature preprocessing step"""
        # First handle leakage detection if not done
        if not self.session.get('leakage_detection_complete'):
            if not self._handle_leakage_detection():
                return False
        
        # Then handle categorical features if not done
        if not self.session.get('categorical_features_complete'):
            if not self._handle_categorical_features():
                return False
        
        # If complete, mark step as done
        if not self.session.get('step_3_complete'):
            self._save_step_summary()
            self.session.set('step_3_complete', True)
            
        return True
        
    def _handle_leakage_detection(self):
        """Handle leakage detection analysis"""
        try:
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            target_col = mappings.get('target')
            
            with self.view.display_spinner('🔍 Analyzing potential target leakage...'):
                task_data = {
                    'df': df,
                    'target_column': target_col
                }
                
                leakage_analysis = self.leakage_detector.execute_task(
                    Task("Detect target leakage", data=task_data)
                )
            
            # Get all features that need attention
            severe_features = leakage_analysis['severe_leakage']
            review_features = [rec['feature'] for rec in leakage_analysis['recommendations']['review']]
            
            if severe_features or review_features:
                warning_msg = "⚠️ Target Leakage Analysis:\n\n"
                
                # Add high-risk features with details
                if severe_features:
                    warning_msg += f"{len(severe_features)} High-risk features detected ⚠️\n"
                    for rec in leakage_analysis['recommendations']['remove']:
                        warning_msg += f"- {rec['feature']}: {rec['reason']}\n"
                    warning_msg += "\n"
                
                # Add potential concern features with details
                if review_features:
                    warning_msg += f"{len(review_features)} Features with potential concerns detected ℹ️\n"
                    for rec in leakage_analysis['recommendations']['review']:
                        warning_msg += f"- {rec['feature']}: {rec['reason']}\n"
                
                self.view.show_message(warning_msg, "error")
                
                # Get mapped columns to exclude from selection
                mapped_columns = set(mappings.values())
                
                # Filter out mapped columns from available features
                available_features = [col for col in df.columns if col not in mapped_columns]
                default_selected = [f for f in (severe_features + review_features) if f not in mapped_columns]
                
                # Let user choose features to remove with all flagged features pre-selected
                features_to_remove = self.view.multiselect(
                    "Select features to remove (high-risk and potential concern features are pre-selected):",
                    options=available_features,
                    default=default_selected
                )
                
                if self.view.display_button("Confirm Feature Removal"):
                    if features_to_remove:
                        # Store removed features in session
                        self.session.set('removed_features', features_to_remove)
                        # Remove selected features from DataFrame
                        df = df.drop(columns=features_to_remove)
                        self.session.set('df', df)
                        
                        self.view.show_message(
                            f"✅ Removed {len(features_to_remove)} features",
                            "success"
                        )
                    return True
                return False
            else:
                self.view.show_message("✅ No target leakage detected", "success")
                return True
            
        except Exception as e:
            self.view.show_message(f"Error in leakage detection: {str(e)}", "error")
            return False
        
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
        removed_features = self.session.get('removed_features', [])
        feature_info = self.session.get('feature_info', {})
        
        summary = "✅ Feature Preprocessing Complete:\n"
        if removed_features:
            summary += f"\n🔍 Removed {len(removed_features)} features due to target leakage:\n"
            for feature in removed_features:
                summary += f"- {feature}\n"
        
        summary += "\n📊 Encoded Features:\n"
        for feature, info in feature_info.items():
            summary += f"- {feature}: **{info['encoding_type']}**\n"
            
        self.session.set('step_3_summary', summary) 