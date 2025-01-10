from sfn_blueprint import Task, SFNValidateAndRetryAgent
from classification_agent.agents.data_splitting_agent import SFNDataSplittingAgent
from classification_agent.config.model_config import DEFAULT_LLM_PROVIDER

class DataSplitting:
    def __init__(self, session_manager, view, validation_window=3):
        self.session = session_manager
        self.view = view
        self.validation_window = validation_window
        self.splitting_agent = SFNDataSplittingAgent(validation_window=validation_window)
        
    def execute(self):
        """Orchestrates the data splitting step"""
    # try:
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        print(f"Available mappings: {mappings}")
        date_col = mappings.get('date')
        print(f">>>>>Date column: {date_col}")
        print(f">>>>>Mappings: {mappings}")
        with self.view.display_spinner('🤖 AI is determining optimal split...'):
            task = Task("Split data", data={
                'df': df, 
                'date_column': date_col,
                'validation_window': self.validation_window,
                'field_mappings': mappings
            })
            validation_task = Task("Validate data splitting", data={
                'df': df, 
                'date_column': date_col,
                'mappings': mappings
            })
            
            validate_and_retry_agent = SFNValidateAndRetryAgent(
                llm_provider=DEFAULT_LLM_PROVIDER,
                for_agent='data_splitter'
            )
            
            split_info, validation_message, is_valid = validate_and_retry_agent.complete(
                agent_to_validate=self.splitting_agent,
                task=task,
                validation_task=validation_task,
                method_name='execute_task',
                get_validation_params='get_validation_params',
                max_retries=2,
                retry_delay=3.0
            )
            
            if not is_valid:
                self.view.show_message("❌ AI couldn't validate data splitting.", "error")
                self.view.show_message(validation_message, "warning")
                return False
            
            # Display split information
            self._display_split_info(split_info)
            
            if self.view.display_button("Confirm Data Split"):
                self.session.set('split_info', split_info)
                self._save_step_summary(split_info)
                self.session.set('step_4_complete', True)
                return True
                
    # except Exception as e:
    #     self.view.show_message(f"Error in data splitting: {str(e)}", "error")
    # return False
        
    def _display_split_info(self, split_info):
        """Display data split information"""
        self.view.display_subheader("Data Split Information")
        
        # Only display date ranges if date information is available
        if split_info['train_start'] is not None:
            self.view.display_markdown("**Training Period:**")
            self.view.display_markdown(f"- Start: {split_info['train_start']}")
            self.view.display_markdown(f"- End: {split_info['train_end']}")
            
            self.view.display_markdown("**Validation Period:**")
            self.view.display_markdown(f"- Start: {split_info['valid_start']}")
            self.view.display_markdown(f"- End: {split_info['valid_end']}")
            
            self.view.display_markdown("**Inference Period:**")
            self.view.display_markdown(f"- Month: {split_info['infer_month']}")
        
        # Display sample counts (always show this)
        self.view.display_markdown("\n**Sample Counts:**")
        self.view.display_markdown(f"- Training: **{split_info['train_samples']}**")
        self.view.display_markdown(f"- Validation: **{split_info['valid_samples']}**")
        self.view.display_markdown(f"- Inference: **{split_info['infer_samples']}**")
        
    def _save_step_summary(self, split_info):
        """Save step summary for display in completed steps"""
        summary = "✅ Data Splitting Complete:\n"
        # Add date range info only if available
        if split_info['train_start'] is not None:
            summary += f"- Period: {split_info['train_start']} to {split_info['train_end']}\n"
        summary += f"- Training samples: **{split_info['train_samples']}**\n"
        summary += f"- Validation samples: **{split_info['valid_samples']}**\n"
        summary += f"- Inference samples: **{split_info['infer_samples']}**"
        self.session.set('step_4_summary', summary) 