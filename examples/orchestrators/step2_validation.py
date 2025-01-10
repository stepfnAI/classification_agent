from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.mapping_agent import SFNMappingAgent as MappingAgent
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER
import pandas as pd
from datetime import datetime

class DataValidation:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.mapping_agent = MappingAgent()
        
    def execute(self):
        """Orchestrates the data validation step"""
        # Handle mapping if not done
        if not self.session.get('mapping_complete'):
            if not self._handle_mapping():
                return False
                
        # Handle data types if not done
        if not self.session.get('data_types_complete'):
            if not self._handle_data_types():
                return False
        
        # If both are complete, mark step as done
        if not self.session.get('step_2_complete'):
            self._save_step_summary()
            self.session.set('step_2_complete', True)
            
        return True
        
    def _handle_mapping(self):
        """Handle column mapping logic"""
        try:
            df = self.session.get('df')
            
            # Get AI suggestions for mapping
            if not self.session.get('suggested_mappings'):
                with self.view.display_spinner('🤖 AI is mapping critical fields...'):
                    mapping_task = Task("Map columns", data=df)
                    validation_task = Task("Validate field mapping", data=df)
                    
                    validate_and_retry_agent = SFNValidateAndRetryAgent(
                        llm_provider=DEFAULT_LLM_PROVIDER,
                        for_agent='field_mapper'
                    )
                    
                    mappings, validation_message, is_valid = validate_and_retry_agent.complete(
                        agent_to_validate=self.mapping_agent,
                        task=mapping_task,
                        validation_task=validation_task,
                        method_name='execute_task',
                        get_validation_params='get_validation_params',
                        max_retries=2,
                        retry_delay=3.0
                    )
                    
                    if is_valid:
                        self.session.set('suggested_mappings', mappings)
                    else:
                        self.view.show_message("❌ AI couldn't generate valid field mappings.", "error")
                        self.view.show_message(validation_message, "warning")
                        return False
            
            # Display mapping interface
            return self._display_mapping_interface()
                
        except Exception as e:
            self.view.show_message(f"Error in mapping: {str(e)}", "error")
            return False
        
    def _display_mapping_interface(self):
        """Display interface for verifying and modifying field mappings"""
        self.view.display_subheader("AI Suggested Critical Field Mappings")
        
        # Get current mappings and available columns
        suggested_mappings = self.session.get('suggested_mappings')
        current_mappings = self.session.get('field_mappings', suggested_mappings)
        df = self.session.get('df')
        all_columns = list(df.columns)
        
        # Check for mandatory fields in AI suggestions
        mandatory_fields = ['target']  # Add other mandatory fields if needed
        missing_mandatory = [field for field in mandatory_fields 
                            if field not in suggested_mappings or not suggested_mappings[field]]
        
        if missing_mandatory:
            missing_fields = ', '.join(missing_mandatory)
            self.view.show_message(
                f"⚠️ AI couldn't find mapping for mandatory field(s): {missing_fields}. "
                "Please map these fields manually.",
                "warning"
            )
            # Automatically switch to manual mapping if mandatory fields are missing
            return self._handle_manual_mapping(all_columns, current_mappings)
        
        # Format message with consistent field names
        message = "🎯 AI Suggested Mappings:\n"
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        for field, mapped_col in suggested_mappings.items():
            display_name = field_display_names.get(field, field)
            message += f"- {display_name}:  **{mapped_col or 'Not Found'}**\n"
        
        self.view.show_message(message, "info")
        self.view.display_markdown("---")
        
        # Show options to proceed
        action = self.view.radio_select(
            "How would you like to proceed?",
            options=[
                "Use AI Recommended Mappings",
                "Select Columns Manually"
            ],
            key="mapping_choice"
        )
        
        if action == "Use AI Recommended Mappings":
            if self.view.display_button("Confirm Mappings"):
                self.session.set('field_mappings', suggested_mappings)
                self.session.set('mapping_complete', True)
                return True
            
        else:  # Select Columns Manually
            return self._handle_manual_mapping(all_columns, current_mappings)
    
    def _handle_manual_mapping(self, all_columns, current_mappings):
        """Handle manual column mapping selection"""
        # Required fields with standardized names
        required_fields = ["cust_id", "target", "revenue"]
        # Changed from product_id to prod_id
        optional_fields = ["date", "prod_id"]  
        
        modified_mappings = {}
        suggested_mappings = self.session.get('suggested_mappings', {})
        
        # Handle required fields
        self.view.display_subheader("Required Fields")
        
        # Define display names for better UI presentation
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        for field in required_fields:
            current_value = suggested_mappings.get(field) or current_mappings.get(field)
            default_index = (all_columns.index(current_value) + 1 
                            if current_value in all_columns 
                            else 0)
            
            modified_mappings[field] = self.view.select_box(
                f"Select column for {field_display_names[field]}",
                options=[""] + all_columns,
                index=default_index
            )
        
        # Handle optional fields
        self.view.display_subheader("Optional Fields")
        for field in optional_fields:
            current_value = suggested_mappings.get(field) or current_mappings.get(field)
            default_index = (all_columns.index(current_value) + 1 
                            if current_value in all_columns 
                            else 0)
            
            value = self.view.select_box(
                f"Select column for {field_display_names[field]} (optional)",
                options=[""] + all_columns,
                index=default_index
            )
            if value:  # Only add if a column was selected
                modified_mappings[field] = value
        
        # Confirm modified mappings
        if self.view.display_button("Confirm Modified Mappings"):
            # Validate that required fields are mapped
            missing_required = [f for f in required_fields if not modified_mappings.get(f)]
            if missing_required:
                self.view.show_message(
                    f"❌ Please map required fields: {', '.join(missing_required)}", 
                    "error"
                )
            else:
                self.session.set('field_mappings', modified_mappings)
                print(f">>>>>Field mappings: {modified_mappings}")
                self.session.set('mapping_complete', True)
                return True
        
        return False
        
    def _handle_data_types(self):
        """Handle data type conversions"""
        try:
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            
            with self.view.display_spinner('Converting data types...'):
                # Make a copy to avoid modifying original
                modified_df = df.copy()
                
                # 1. Convert ID column to text
                id_col = mappings.get('CUST_ID')
                if id_col:
                    modified_df[id_col] = modified_df[id_col].astype(str)
                
                # 2. Convert date column to YYYY-MM format
                date_col = mappings.get('BILLING_DATE')
                if date_col:
                    try:
                        # First try to parse dates
                        modified_df[date_col] = pd.to_datetime(modified_df[date_col])
                        # Then convert to YYYY-MM format
                        modified_df[date_col] = modified_df[date_col].dt.strftime('%Y-%m')
                    except Exception as e:
                        self.view.show_message(f"Error converting date column: {str(e)}", "error")
                        return False
                
                # 3. Check target values (if present)
                target_col = mappings.get('TARGET')
                if target_col:
                    unique_values = modified_df[target_col].nunique()
                    if unique_values != 2:
                        self.view.show_message(
                            f"⚠️ Warning: Target column should have exactly 2 unique values, but found {unique_values}",
                            "warning"
                        )
            
            # Display modified data for confirmation
            self.view.display_subheader("Modified Data Types")
            self.view.display_dataframe(modified_df.head())
            
            # Show data type information
            self.view.display_subheader("Data Types Summary")
            dtypes_msg = "Updated Data Types:\n"
            for col, dtype in modified_df.dtypes.items():
                if col in mappings.values():
                    dtypes_msg += f"- {col}: **{dtype}**\n"
            self.view.show_message(dtypes_msg, "info")
            
            if self.view.display_button("Confirm Data Types"):
                self.session.set('df', modified_df)
                self.session.set('data_types_complete', True)
                return True
                
        except Exception as e:
            self.view.show_message(f"Error in data type conversion: {str(e)}", "error")
        return False
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        mappings = self.session.get('field_mappings')
        df = self.session.get('df')
        
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        summary = "✅ Data Validation Complete:\n"
        for field, col in mappings.items():
            dtype = df[col].dtype if col else None
            display_name = field_display_names.get(field, field)
            summary += f"- {display_name}: **{col}** ({dtype})\n"
        self.session.set('step_2_summary', summary) 