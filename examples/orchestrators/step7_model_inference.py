from sfn_blueprint import Task
from classification_agent.utils.model_manager import ModelManager
import pandas as pd

class ModelInference:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.model_manager = ModelManager()
        
    def execute(self):
        """Orchestrates the model inference step"""
        # Get split info which contains inference data
        split_info = self.session.get('split_info')
        if split_info is None or 'infer_df' not in split_info:
            self.view.show_message("❌ No inference data found in split info.", "error")
            return False
            
        # Get inference data from split info
        inference_data = split_info['infer_df']
            
        # Get selected model info
        model_results = self.session.get('model_results', {})
        selected_model = self.session.get('selected_model')
        if not selected_model or selected_model not in model_results:
            self.view.show_message("❌ No model selected.", "error")
            return False
            
        # Get model ID from results
        model_id = model_results[selected_model].get('model_id')
        if not model_id:
            self.view.show_message("❌ Model ID not found.", "error")
            return False
            
        # Display inference button if not already done
        if not self.session.get('step_7_complete'):
            self.view.display_subheader("🔮 Model Inference")
            if self.view.display_button("Run Inference", key="run_inference"):
                try:
                    # Load the model and metadata
                    model, metadata = self.model_manager.load_model(model_id)
                    
                    # Get features from metadata
                    train_features = metadata.get('features', [])
                    if not train_features:
                        self.view.show_message("❌ Training features not found in model metadata.", "error")
                        return False
                    
                    # Prepare inference data using same features as training
                    X_inference = inference_data[train_features]
                    
                    # Make predictions
                    predictions = model.predict_proba(X_inference)[:, 1]
                    
                    # Add predictions to inference data
                    results_df = inference_data.copy()
                    results_df['predicted_probability'] = predictions
                    results_df['predicted_class'] = (predictions >= 0.5).astype(int)
                    
                    # Save results
                    self.session.set('inference_results', results_df)
                    
                    # Display results summary
                    self._display_results_summary(results_df)
                    
                    # Save step summary
                    self._save_step_summary(results_df)
                    
                    self.session.set('step_7_complete', True)
                except Exception as e:
                    self.view.show_message(f"❌ Error during inference: {str(e)}", "error")
                    return False
        
        # Always show results if available (moved outside the if block)
        results_df = self.session.get('inference_results')
        if results_df is not None:
            # Show summary
            summary = self.session.get('step_7_summary')
            if summary:
                self.view.show_message(summary, "success")
            
            # Display full results
            self.view.display_subheader("📊 Detailed Results")
            self.view.display_dataframe(results_df)
            
            # Add download button
            self.view.create_download_button(
                label="📥 Download Predictions as CSV",
                data=results_df.to_csv(index=False),
                file_name="prediction_results.csv",
                mime_type='text/csv'
            )
        
        # Return True only if step is complete
        return self.session.get('step_7_complete', False)
        
    def _display_results_summary(self, results_df):
        """Display summary of inference results"""
        total_records = len(results_df)
        positive_preds = (results_df['predicted_class'] == 1).sum()
        avg_probability = results_df['predicted_probability'].mean()
        
        self.view.show_message(
            f"**Inference Results Summary:**\n"
            f"- Total Records: {total_records}\n"
            f"- Positive Predictions: {positive_preds} ({(positive_preds/total_records)*100:.1f}%)\n"
            f"- Average Probability: {avg_probability:.3f}",
            "info"
        )
        
        # Display sample of results
        self.view.display_subheader("Sample Predictions")
        self.view.display_dataframe(results_df.head())
        
    def _save_step_summary(self, results_df):
        """Save step summary for display in completed steps"""
        total_records = len(results_df)
        positive_preds = (results_df['predicted_class'] == 1).sum()
        
        summary = "✅ Model Inference Complete:\n"
        summary += f"- Processed Records: **{total_records}**\n"
        summary += f"- Positive Predictions: **{positive_preds}** ({(positive_preds/total_records)*100:.1f}%)\n"
        
        self.session.set('step_7_summary', summary) 