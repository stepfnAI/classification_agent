from sfn_blueprint import MODEL_CONFIG
DEFAULT_LLM_PROVIDER = 'openai'
DEFAULT_LLM_MODEL = 'gpt-4o-mini'

MODEL_CONFIG["mapping_agent"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 300,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["data_type_suggester"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["code_generator"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["categorical_feature_handler"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["model_trainer"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["model_selector"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["data_splitter"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["leakage_detector"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
} 