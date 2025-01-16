from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.mapping_agent import SFNMappingAgent as MappingAgent
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class DataValidation:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.mapping_agent = MappingAgent()
        self.excluded_features = {}  # Will store features excluded from model training
        
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
            mapping_confirmed = self.session.get('mapping_confirmed', False)
            
            # Get AI suggestions for mapping only if not already confirmed
            if not mapping_confirmed and not self.session.get('suggested_mappings'):
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
            result = self._display_mapping_interface()
            
            # If mapping is confirmed, don't show the interface again
            if result:
                self.session.set('mapping_confirmed', True)
            
            return result
                
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
        mapping_confirmed = self.session.get('mapping_confirmed', False)
        
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
        
        # Only show options if mapping not yet confirmed
        if not mapping_confirmed:
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
                    self.session.set('mapping_confirmed', True)  # Set confirmation flag
                    return True
                    
            else:  # Select Columns Manually
                return self._handle_manual_mapping(all_columns, current_mappings)
        
        return False
        
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
                self.session.set('mapping_confirmed', True)  # Set confirmation flag
                return True
        
        return False
        
    def _handle_data_types(self):
        """Handle data type conversions"""
        # try:
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        target_processed = self.session.get('target_classification_complete', False)
        
        with self.view.display_spinner('Converting data types...'):
            # Make a copy to avoid modifying original
            modified_df = df.copy()
            
            # 1. Convert ID column to text
            id_col = mappings.get('cust_id')
            if id_col:
                modified_df[id_col] = modified_df[id_col].astype(str)
            
            # 2. Convert date column to YYYY-MM format
            date_col = mappings.get('date')
            if date_col:
                try:
                    modified_df[date_col] = pd.to_datetime(modified_df[date_col])
                    modified_df[date_col] = modified_df[date_col].dt.strftime('%Y-%m')
                except Exception as e:
                    self.view.show_message(f"Error converting date column: {str(e)}", "error")
                    return False
            
            # 3. Check target values (if present)
            target_col = mappings.get('target')
            if target_col and not target_processed:  # Only process if not already done
                target_handling_result = self._handle_target_classification(modified_df, target_col)
                if isinstance(target_handling_result, bool) and not target_handling_result:
                    return False
                modified_df = target_handling_result
                # Update session DataFrame immediately after target classification
                self.session.set('df', modified_df)

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
            self.session.set('df', modified_df)  # Update again to be safe
            self.session.set('data_types_complete', True)
            return True
                
        # except Exception as e:
        #     self.view.show_message(f"Error in data type conversion: {str(e)}", "error")
        # return False
        
    def _handle_target_classification(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Handle target column classification"""
        # Get unique values excluding NaN/None/null
        unique_values = sorted([v for v in df[target_col].unique() 
                              if pd.notna(v) and v not in ['', 'None', 'null']])
        n_unique = len(unique_values)
        
        # If more than 6 unique values, suggest choosing another column
        if n_unique > 6:
            self.view.show_message(
                f"⚠️ Target column '{target_col}' has {n_unique} unique values, which is too many. "
                "Please select another column that can be used as a binary target.",
                "error"
            )
            return False
        
        # Show value distribution (including NaN for information)
        value_counts = df[target_col].value_counts(dropna=False)
        dist_msg = f"Value distribution in '{target_col}':\n"
        
        # First show non-null values
        for val in unique_values:
            count = value_counts.get(val, 0)
            percentage = (count / len(df)) * 100
            dist_msg += f"- {val}: {count} records ({percentage:.1f}%)\n"
        
        # Then show null/NaN count if any (for information only)
        null_count = df[target_col].isna().sum()
        if null_count > 0:
            percentage = (null_count / len(df)) * 100
            dist_msg += f"- Missing/NaN: {null_count} records ({percentage:.1f}%) - Will be used for inference\n"
        
        self.view.show_message(dist_msg, "info")

        # For binary case (2 unique values)
        if n_unique == 2:
            # Check if values are already 0 and 1
            if set(unique_values) == {0, 1}:
                # Automatically use existing 0/1 encoding
                self.view.show_message("✅ Target column already contains binary values (0/1)", "info")
                self.session.set('target_classification_complete', True)
                return df
            else:
                # For non-numeric binary values, let user choose
                self.view.display_subheader("Binary Target Classification")
                class_0_value = self.view.select_box(
                    "Select the value that should represent Class 0 (Negative):",
                    options=unique_values
                )
                
                if class_0_value:
                    class_1_value = [v for v in unique_values if v != class_0_value][0]
                    preview_msg = "Preview of classification:\n"
                    preview_msg += f"Class 0: {class_0_value} ({len(df[df[target_col] == class_0_value])} records)\n"
                    preview_msg += f"Class 1: {class_1_value} ({len(df[df[target_col] == class_1_value])} records)\n"
                    self.view.show_message(preview_msg, "info")
                    
                    if self.view.display_button("Confirm Classification"):
                        return self._apply_target_encoding(df, target_col, 
                                                        class_0_values=[class_0_value],
                                                        class_1_values=[class_1_value])
                return False
        
        # For 3-6 unique values, let user classify them
        self.view.display_subheader("Target Value Classification")
        
        # Create side-by-side columns using view method
        left_col, right_col = self.view.create_columns([1, 1])
        
        # Left column for class 0
        with left_col:
            self.view.display_subheader("Class 0 (Negative)")
            class_0_values = self.view.multiselect(
                "Select values for class 0:",
                options=unique_values,
                key="class_0_selection"
            )
        
        # Right column for class 1
        with right_col:
            self.view.display_subheader("Class 1 (Positive)")
            remaining_values = [v for v in unique_values if v not in class_0_values]
            class_1_values = self.view.multiselect(
                "Select values for class 1:",
                options=unique_values,
                key="class_1_selection"
            )
        
        # Validation messages
        validation_error = None
        if not class_0_values and not class_1_values:
            validation_error = "Please select values for at least one class"
        elif any(v in class_0_values and v in class_1_values for v in unique_values):
            validation_error = "A value cannot be in both classes"
        elif len(set(class_0_values + class_1_values)) != n_unique:
            validation_error = "All values must be classified"
        
        if validation_error:
            self.view.show_message(f"❌ {validation_error}", "error")
        else:
            # Show preview of classification
            preview_msg = "Preview of classification:\n"
            preview_msg += f"Class 0: {sorted(class_0_values)} ({len(df[df[target_col].isin(class_0_values)])} records)\n"
            preview_msg += f"Class 1: {sorted(class_1_values)} ({len(df[df[target_col].isin(class_1_values)])} records)\n"
            self.view.show_message(preview_msg, "info")
            
            # Add confirmation button
            if self.view.display_button("Confirm Classification"):
                return self._apply_target_encoding(df, target_col, class_0_values, class_1_values)
        
        return False
        
    def _apply_target_encoding(self, df: pd.DataFrame, target_col: str, 
                              class_0_values: List, class_1_values: List) -> pd.DataFrame:
        """Apply target encoding and update session data"""
        # Create new binary target column
        new_target_col = f"{target_col}_binary"
        df[new_target_col] = df[target_col].isin(class_1_values).astype(int)
        
        # Store original target column info in excluded features
        self.excluded_features['target_original'] = {
            'column': target_col,
            'mapping': {
                0: class_0_values,
                1: class_1_values
            }
        }
        
        # Remove original target column
        df = df.drop(columns=[target_col])
        
        # Update mappings to use new target column
        mappings = self.session.get('field_mappings')
        mappings['target'] = new_target_col
        self.session.set('field_mappings', mappings)
        # Mark target classification as complete
        self.session.set('target_classification_complete', True)
        
        return df
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        mappings = self.session.get('field_mappings')
        df = self.session.get('df')
        print(f">>>>>Field mappings1: {mappings}")
        print(f'>>>df1: {df.head(3)}')
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