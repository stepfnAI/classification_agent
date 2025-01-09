import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any
import shutil

class ModelManager:
    def __init__(self):
        # Create base paths
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.models_dir = os.path.join(self.base_dir, 'models', 'trained')
        self.registry_path = os.path.join(self.base_dir, 'models', 'registry.json')
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)  # Create models directory if it doesn't exist
        
        # Initialize registry if it doesn't exist
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump({}, f)

    def save_model(self, model: Any, model_name: str, metadata: Dict) -> str:
        """Save model and its metadata"""
        # Generate unique model ID using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{model_name}_{timestamp}"
        
        # Create paths
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Update metadata with model info
        metadata.update({
            'model_id': model_id,
            'model_name': model_name,
            'created_at': timestamp,
            'model_path': model_path
        })
        
        # Update registry
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)
        
        registry[model_id] = metadata
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
        return model_id

    def load_model(self, model_id):
        """Load a model from the registry
        
        Args:
            model_id: ID of the model to load
        Returns:
            tuple: (loaded_model, metadata)
        """
        try:
            # Check if registry file exists
            if not os.path.exists(self.registry_path):
                raise ValueError("Model registry not found")
                
            # Load registry
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            
            if model_id not in registry:
                raise ValueError(f"Model {model_id} not found in registry")
                
            metadata = registry[model_id]
            
            # Check if model file exists
            if not os.path.exists(metadata['model_path']):
                raise ValueError(f"Model file not found at {metadata['model_path']}")
                
            model = joblib.load(metadata['model_path'])
            
            return model, metadata
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def cleanup(self):
        """Remove all saved models and reset registry"""
        try:
            # Remove all files in models directory
            if os.path.exists(self.models_dir):
                shutil.rmtree(self.models_dir)
                os.makedirs(self.models_dir)  # Recreate empty directory
            
            # Reset registry
            with open(self.registry_path, 'w') as f:
                json.dump({}, f)
                
            return True
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            return False 