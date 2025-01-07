from sfn_blueprint.views.streamlit_view import SFNStreamlitView
from typing import Any, List, Optional
import streamlit as st
import os
from pathlib import Path

class StreamlitView(SFNStreamlitView):
    def __init__(self, title: str):
        self.title = title

    def file_uploader(self, label: str, accepted_types: List[str], key: Optional[str] = None) -> Any:
        """Override file_uploader to include key parameter"""
        return st.file_uploader(label, type=accepted_types, key=key)

    def select_box(self, label: str, options: List[str], index: Optional[int] = None) -> str:
        """Add select box functionality"""
        return st.selectbox(label, options, index=index)
    
    def save_uploaded_file(self, uploaded_file: Any) -> Optional[str]:
        """Save uploaded file temporarily"""
        if uploaded_file is not None:
            # Create temp directory if not exists
            temp_dir = Path('./temp_files')
            temp_dir.mkdir(exist_ok=True)

            # Save file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return str(file_path)
        return None
    
    def delete_uploaded_file(self, file_path: str) -> bool:
        """Delete temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            self.show_message(f"Error deleting file: {e}", "error")
        return False 