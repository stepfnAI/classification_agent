from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.model_selection_agent import SFNModelSelectionAgent
from classification_agent.utils.model_manager import ModelManager
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER

class ModelSelection:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.selection_agent = SFNModelSelectionAgent()
        self.model_manager = ModelManager()
        
    def execute(self):
        """Orchestrates the model selection step"""
        # Get model results
        model_results = self.session.get('model_results')
        
        if not model_results:
            self.view.show_message("❌ No trained models found.", "error")
            return False
            
        # Get model recommendation if not done
        if not self.session.get('model_recommended'):
            validate_and_retry_agent = SFNValidateAndRetryAgent(
                llm_provider=DEFAULT_LLM_PROVIDER,
                for_agent='model_selector'
            )
            
            task = Task("Select best model", data={
                'model_results': model_results,
                'custom_instructions': ""
            })
            
            result, validation_message, is_valid = validate_and_retry_agent.complete(
                agent_to_validate=self.selection_agent,
                task=task,
                validation_task=task,
                method_name='execute_task',
                get_validation_params='get_validation_params',
                max_retries=2,
                retry_delay=3.0
            )
            
            if not is_valid:
                self.view.show_message(f"❌ Model selection failed: {validation_message}", "error")
                return False
                
            self.session.set('model_recommendation', result)
            self.session.set('model_recommended', True)
        
        # Get the recommendation from session
        recommendation = self.session.get('model_recommendation')
        
        # Always display recommendation if we have it
        if recommendation:
            self._display_recommendation(recommendation)
        
        # If we haven't completed selection yet, show selection options
        if not self.session.get('step_6_complete'):
            # Get user confirmation
            selected_model = self._get_user_selection(model_results.keys(), recommendation['selected_model'])
            if selected_model:
                self.session.set('selected_model', selected_model)
                self.session.set('selection_mode', 'ai' if selected_model == recommendation['selected_model'] else 'manual')
                self._save_step_summary()
                self.session.set('step_6_complete', True)
                return True
            return False
        
        # If we've already completed, show summary
        summary = self.session.get('step_6_summary')
        if summary:
            self.view.show_message(summary, "success")
        
        return True
        
    def _display_recommendation(self, recommendation):
        """Display model recommendation and explanation"""
        self.view.display_subheader("🤖 Model Recommendation")
        
        # Show selected model
        self.view.show_message(f"**Recommended Model:** {recommendation['selected_model']}", "success")
        
        # Show explanation
        self.view.show_message("**Explanation:**\n" + recommendation['explanation'], "info")
        
        # Show comparison summary
        self.view.show_message("**Model Comparison:**\n" + recommendation['comparison_summary'], "info")
        
    def _get_user_selection(self, available_models, recommended_model):
        """Get user confirmation or alternative selection"""
        self.view.display_subheader("Model Selection")
        
        # First let user choose the selection mode
        selection_mode = self.view.display_radio(
            "How would you like to select the model?",
            options=["Proceed with AI Recommendation", "Select Model Manually"],
            key="selection_mode"
        )
        
        if selection_mode == "Proceed with AI Recommendation":
            # If using AI recommendation, just show confirmation
            if self.view.display_button("✅ Confirm AI Recommendation", key="confirm_ai"):
                return recommended_model
        else:
            # If manual selection, show all models
            selected = self.view.display_radio(
                "Select model to use:",
                options=list(available_models),
                key="model_selection"
            )
            
            if self.view.display_button("✅ Confirm Selection", key="confirm_manual"):
                if selected:
                    return selected
                else:
                    self.view.show_message("⚠️ Please select a model first", "warning")
                    return None
        
        return None
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        selected_model = self.session.get('selected_model')
        selection_mode = self.session.get('selection_mode')
        
        if not selected_model:  # Safety check
            return
            
        summary = "✅ Model Selection Complete:\n"
        summary += f"- Selected Model: **{selected_model}**\n"
        
        if selection_mode == 'ai':
            summary += "- Using AI recommended model\n"
        else:
            summary += "- Using manually selected model\n"
        
        self.session.set('step_6_summary', summary) 