import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
from orchestrators.step1_data_upload import DataUpload
from orchestrators.step2_validation import DataValidation
from orchestrators.step3_preprocessing import FeaturePreprocessing
from orchestrators.step4_splitting import DataSplitting
from orchestrators.step5_model_training import ModelTraining
from views.streamlit_views import StreamlitView
from sfn_blueprint import SFNSessionManager
from classification_agent.utils.model_manager import ModelManager
from examples.orchestrators.step6_model_selection import ModelSelection
from examples.orchestrators.step7_model_inference import ModelInference

class ClassificationApp:
    def __init__(self):
        self.view = StreamlitView(title="Classification App")
        self.session = SFNSessionManager()
        self.orchestrators = {
            1: DataUpload(self.session, self.view),
            2: DataValidation(self.session, self.view),
            3: FeaturePreprocessing(self.session, self.view),
            4: DataSplitting(self.session, self.view, validation_window=3),
            5: ModelTraining(self.session, self.view),
            6: ModelSelection(self.session, self.view),
            7: ModelInference(self.session, self.view)
        }
        self.step_titles = {
            1: "Data Upload",
            2: "Data Validation",
            3: "Feature Preprocessing",
            4: "Data Splitting",
            5: "Model Training",
            6: "Model Selection",
            7: "Model Inference"
        }
        
    def run(self):
        """Main application flow"""
        self._initialize_ui()
        
        # Get current step
        current_step = self.session.get('current_step', 1)
        
        # Display completed steps
        self._display_completed_steps(current_step)
        
        # Display current step header
        self.view.display_header(f"Step {current_step}: {self.step_titles[current_step]}")
        
        # Execute current step
        if current_step in self.orchestrators:
            self.view.display_markdown("---")
            result = self.orchestrators[current_step].execute()
            
            # Only advance if not the final step
            if result and current_step < len(self.orchestrators):
                self._advance_step()
            # For final step, just stay on the same page
            elif result and current_step == len(self.orchestrators):
                return
    
    def _initialize_ui(self):
        """Initialize the UI components"""
        col1, col2 = self.view.create_columns([7, 1])
        with col1:
            self.view.display_title()
        with col2:
            if self.view.display_button("🔄 Reset", key="reset_button"):
                # Clean up saved models
                model_manager = ModelManager()
                model_manager.cleanup()
                # Clear session and rerun
                self.session.clear()
                self.view.rerun_script()
    
    def _display_completed_steps(self, current_step):
        """Display summary of completed steps"""
        if current_step <= 1:
            return
            
        for step in range(1, current_step):
            if self.session.get(f'step_{step}_complete'):
                self.view.display_header(f"Step {step}: {self.step_titles[step]}")
                self._display_step_summary(step)
                self.view.display_markdown("---")
    
    def _display_step_summary(self, step):
        """Display summary for a completed step"""
        summary = self.session.get(f'step_{step}_summary')
        if summary:
            self.view.show_message(summary, "success")
    
    def _advance_step(self):
        """Advance to the next step"""
        current_step = self.session.get('current_step', 1)
        self.session.set('current_step', current_step + 1)
        self.view.rerun_script()

if __name__ == "__main__":
    app = ClassificationApp()
    app.run() 