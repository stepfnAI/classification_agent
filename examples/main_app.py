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
from views.streamlit_views import StreamlitView
from sfn_blueprint import SFNSessionManager

class ClassificationApp:
    def __init__(self):
        self.view = StreamlitView(title="Classification App")
        self.session = SFNSessionManager()
        self.orchestrators = {
            1: DataUpload(self.session, self.view),
            2: DataValidation(self.session, self.view),
            3: FeaturePreprocessing(self.session, self.view),
            4: DataSplitting(self.session, self.view, validation_window=3)
        }
        self.step_titles = {
            1: "Data Upload",
            2: "Data Validation",
            3: "Feature Preprocessing",
            4: "Data Splitting"
        }
        
    def run(self):
        """Main application flow"""
        # Initialize UI
        self._initialize_ui()
        
        current_step = self.session.get('current_step', 1)
        
        # Display progress of completed steps
        self._display_completed_steps(current_step)
        
        # Execute current step
        if current_step in self.orchestrators:
            self.view.display_header(f"Step {current_step}: {self.step_titles[current_step]}")
            self.view.display_markdown("---")
            
            if self.orchestrators[current_step].execute():
                self._advance_step()
    
    def _initialize_ui(self):
        """Initialize the UI components"""
        col1, col2 = self.view.create_columns(2)
        with col1:
            self.view.display_title()
        with col2:
            if self.view.display_button("🔄 Reset", key="reset_button"):
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
    st.set_page_config(page_title="Classification App", layout="wide")
    app = ClassificationApp()
    app.run() 