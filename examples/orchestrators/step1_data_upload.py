import pandas as pd
from sfn_blueprint import SFNDataLoader, Task

class DataUpload:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.data_loader = SFNDataLoader()
        
    def execute(self):
        """Orchestrates the data upload step"""
        self.view.display_header("Step 1: Data Upload and Preview")
        self.view.display_markdown("---")
        
        # Always show data preview if data is loaded
        df = self.session.get('df')
        if df is not None:
            self.view.display_subheader("Data Preview")
            self.view.display_dataframe(df.head())
        
        # Show file uploader only if no data is loaded
        uploaded_file = self.view.file_uploader(
            "Choose a CSV or Excel file", 
            accepted_types=["csv", "xlsx", "json", "parquet"],
            key="data_upload"
        )

        if uploaded_file is not None:
            with self.view.display_spinner('Loading data...'):
                try:
                    file_path = self.view.save_uploaded_file(uploaded_file)
                    load_task = Task("Load the uploaded file", data=uploaded_file, path=file_path)
                    
                    df = self.data_loader.execute_task(load_task)
                    self.view.delete_uploaded_file(file_path)
                    
                    self.session.set('df', df)
                    self.view.show_message(f"✅ Data loaded successfully. Shape: {df.shape}", "success")
                    self.view.display_subheader("Data Preview")
                    self.view.display_dataframe(df.head())
                    
                    if not self.session.get('step_1_complete'):
                        if self.view.display_button("Confirm and Continue", key="confirm_data"):
                            self._save_step_summary(df)
                            self.session.set('step_1_complete', True)
                            return True
                    
                except Exception as e:
                    self.view.show_message(f"❌ Error loading file: {str(e)}", "error")
        
        return False
    
    def _save_step_summary(self, df):
        """Save step summary for display in completed steps"""
        summary = (f"✅ Data Upload Complete:\n"
                  f"- Rows: **{df.shape[0]}**\n"
                  f"- Columns: **{df.shape[1]}**\n"
                  f"- File loaded successfully")
        self.session.set('step_1_summary', summary) 