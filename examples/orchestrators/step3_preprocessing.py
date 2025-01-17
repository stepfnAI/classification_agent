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
            
            # Display findings
            self.view.display_subheader("🎯 Target Leakage Analysis")
            
            # Handle severe leakage features (correlation > 0.95)
            severe_features = leakage_analysis['severe_leakage']
            if severe_features:
                self.view.show_message(
                    "⚠️ High-Risk Features Detected (Recommended for Removal):", 
                    "warning"
                )
                for rec in leakage_analysis['recommendations']['remove']:
                    self.view.show_message(
                        f"- **{rec['feature']}**\n  {rec['reason']}",
                        "warning"
                    )
                
                # Let user choose which features to remove
                features_to_remove = self.view.multiselect(
                    "Select features to remove due to target leakage:",
                    severe_features
                )
                
                if features_to_remove:
                    # Store removed features in session
                    self.session.set('removed_features', features_to_remove)
                    # Remove selected features from DataFrame
                    df = df.drop(columns=features_to_remove)
                    self.session.set('df', df)
                    
                    self.view.show_message(
                        f"✅ Removed {len(features_to_remove)} features with severe target leakage",
                        "success"
                    )
            else:
                self.view.show_message("✅ No severe target leakage detected", "success")
            
            # Handle suspicious features (correlation 0.90-0.95 or other concerns)
            review_features = leakage_analysis['recommendations']['review']
            if review_features:
                self.view.show_message(
                    "ℹ️ Features to Review (Potential Concerns):", 
                    "info"
                )
                for rec in review_features:
                    self.view.show_message(
                        f"- **{rec['feature']}**\n  {rec['reason']}",
                        "info"
                    )
            
            # Display detailed analysis if requested
            if self.view.display_button("Show Detailed Analysis"):
                analysis_msg = "\n📊 Detailed Feature Analysis:\n"
                for feature, metrics in leakage_analysis['analysis'].items():
                    analysis_msg += (
                        f"\n**{feature}**:\n"
                        f"- Correlation: {metrics['correlation']:.3f}\n"
                        f"- Missing Values: {metrics['null_percentage']:.1%}\n"
                        f"- Unique Ratio: {metrics['unique_ratio']:.2f}\n"
                    )
                self.view.show_message(analysis_msg, "info")
            
            self.session.set('leakage_detection_complete', True)
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