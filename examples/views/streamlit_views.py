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

    def display_radio(self, label: str, options: list, key: str = None) -> str:
        """Display a radio button group
        
        Args:
            label (str): Label for the radio group
            options (list): List of options to display
            key (str): Unique key for the component
            
        Returns:
            str: Selected option
        """
        return st.radio(label, options, key=key) 

    def text_area(self, label: str, value: str = "", height: int = None, help: str = None, key: str = None) -> str:
        """Display a multi-line text input widget
        
        Args:
            label (str): Label for the text area
            value (str, optional): Default text value. Defaults to "".
            height (int, optional): Height of the text area in pixels. Defaults to None.
            help (str, optional): Tooltip help text. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            str: Text entered by the user
        """
        return st.text_area(
            label=label,
            value=value,
            height=height,
            help=help,
            key=key
        ) 

    def multiselect(self, label: str, options: List[Any], default: Optional[List[Any]] = None, key: Optional[str] = None) -> List[Any]:
        """Display a multi-select widget
        
        Args:
            label (str): Label for the multi-select
            options (List[Any]): List of options to choose from
            default (List[Any], optional): Default selected values. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            List[Any]: List of selected options
        """
        return st.multiselect(
            label=label,
            options=options,
            default=default,
            key=key
        ) 

    def create_columns(self, ratios: List[int]) -> tuple:
        """Create columns with specified width ratios
        
        Args:
            ratios (List[int]): List of relative widths for columns
            
        Returns:
            tuple: Tuple of column objects
        """
        return st.columns(ratios)

    def get_column(self, index: int):
        """Get a specific column
        
        Args:
            index (int): Index of the column to get
            
        Returns:
            streamlit.delta_generator.DeltaGenerator: Column object
        """
        return self.columns[index] 