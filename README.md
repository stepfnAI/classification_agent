# Classification Agent

An AI-powered classification agent that automates the end-to-end process of building and selecting binary classification models, with interactive capabilities and comprehensive model analysis.

## 🌟 Features

- **Binary Classification Support**: Currently optimized for binary classification problems only
- **Intelligent Data Processing**: Automatically handles data preprocessing and validation
- **Smart Feature Engineering**: 
  - AI-powered categorical feature handling and encoding
  - Automated target leakage detection and mitigation
  - Interactive feature selection and removal
- **Automated Model Pipeline**:
  - Data validation and cleaning
  - Feature preprocessing
  - Intelligent data splitting
  - Model training with multiple algorithms
  - Model selection and evaluation
- **Interactive Interface**: Review and modify AI suggestions at each step
- **Flexible Data Input**: Supports various data formats
- **Visual Progress Tracking**: Clear feedback on each step

## 🚀 Getting Started

### Prerequisites

- Python 3.9-3.11
- OpenAI API key

### Platform-Specific Requirements

#### macOS
If you're using macOS, you'll need to install OpenMP support:

```bash
brew install libomp
```

#### Ubuntu/Debian
Install OpenMP support using:

```
bash
sudo apt-get install libomp-dev
```

#### Windows
No additional steps required - OpenMP support is included with the standard installation.


### Installation

1. Clone the repository:

```
git clone git@github.com:stepfnAI/classification_agent.git
cd classification_agent
```

2. Create and activate a virtual environment:

```bash
pip install virtualenv                # Install virtualenv if not already installed
virtualenv venv                       # Create virtual environment
source venv/bin/activate             # Linux/Mac
# OR
.\venv\Scripts\activate               # Windows
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

### Running the Application

```bash
streamlit run examples/main_app.py
```


## 🔄 Workflow

1. **Data Upload**
   - Upload your dataset
   - Initial data preview
   - Reset functionality available

2. **Data Validation**
   - Automatic field mapping (CUST_ID, REVENUE, TARGET, etc.)
   - Data type validation and suggestions
   - Interactive review and modification

3. **Feature Preprocessing**
   - Automatic categorical feature detection
   - Intelligent encoding selection
   - Interactive preprocessing options

4. **Data Splitting**
   - Smart train/validation/inference split
   - Temporal awareness for time-series data
   - Configurable validation window

5. **Model Training**
   - Multiple model training (LightGBM, CatBoost)
   - Automated hyperparameter optimization
   - Progress tracking

6. **Model Selection**
   - Comprehensive model evaluation
   - AI-powered model recommendations
   - Interactive model selection interface

7. **Model Inference**
   - Batch prediction capabilities
   - Model explanation and insights
   - Results export

## 🛠️ Key Components

- **SFNMappingAgent**: Handles field mapping
- **SFNDataTypeSuggestionAgent**: Manages data type validation
- **SFNCategoricalFeatureAgent**: Processes categorical features
- **SFNDataSplittingAgent**: Manages data splitting
- **SFNModelTrainingAgent**: Handles model training
- **SFNModelSelectionAgent**: Provides model recommendations
- **StreamlitView**: Manages user interface
- **SFNSessionManager**: Handles application state

## 📦 Dependencies

- sfn_blueprint==0.5.2
- sfn-llm-client==0.1.0
- lightgbm>=4.1.0
- catboost>=1.2.2

## 🔒 Security

- Secure API key handling
- Input validation
- Safe data processing
- Environment variable management

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact
puneet@stepfunction.ai