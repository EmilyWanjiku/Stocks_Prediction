# 📈 Kenyan Bank Stock Forecast Dashboard

A comprehensive machine learning-powered web application for forecasting Kenyan bank stock prices using hybrid ensemble models (LSTM, ARIMA, and XGBoost).

## ✨ Features

### 🔐 Authentication System
- **Secure Login/Registration**: File-based authentication with password hashing
- **Password Strength Indicator**: Real-time feedback on password security
- **Session Management**: Persistent login sessions
- **User-Friendly Forms**: Modern, intuitive authentication interface

### 📊 Dashboard Features
- **Real-Time Statistics Cards**: Current price, 52-week high/low, volatility metrics
- **Interactive Visualizations**: 
  - Historical price charts
  - Model comparison charts
  - Future forecast visualizations
  - Accuracy metrics comparison
- **Advanced Metrics Table**: Color-coded performance metrics with best model indicator
- **Forecast Export**: Download forecasts as CSV or Excel files
- **Multi-Bank Support**: Analyze and compare multiple Kenyan banks

### 🤖 Machine Learning Models
- **LSTM (Long Short-Term Memory)**: Deep learning time-series model
- **ARIMA**: Statistical forecasting model
- **XGBoost**: Gradient boosting regression model
- **Hybrid Ensemble**: Weighted combination of all models based on inverse RMSE

### 🎨 User Interface
- **Modern Design**: Clean, professional interface with gradient styling
- **Responsive Layout**: Optimized for different screen sizes
- **Tabbed Navigation**: Organized chart views for better UX
- **Color-Coded Metrics**: Visual indicators for performance metrics
- **Interactive Charts**: Plotly-powered interactive visualizations

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git (for cloning the repository)
- Windows PowerShell (for Windows) or Terminal (for Mac/Linux)

### Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Stocks_Prediction.git
cd Stocks_Prediction
```

Replace `yourusername` with your actual GitHub username.

#### Step 2: Set Up Virtual Environment

**Windows:**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### Step 3: Install Dependencies

**Windows (using PowerShell script):**
```powershell
powershell -ExecutionPolicy Bypass -File install_dependencies.ps1
```

**Mac/Linux (using requirements.txt):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative (manual installation):**
```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn tensorflow statsmodels xgboost joblib matplotlib plotly streamlit openpyxl werkzeug
```

#### Step 4: Prepare Data

1. Place your bank stock CSV files in the `data/` directory
2. Files should follow the naming pattern: `{BANK}_stocks.csv` (e.g., `ABSA_stocks.csv`, `KCB_stocks.csv`)
3. Each CSV should contain columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `StockCode`

#### Step 5: Train the Models

```bash
python train_models.py
```

This will:
- Load data from the `data/` directory
- Train LSTM, ARIMA, and XGBoost models for each bank
- Save trained models to `app/models/`
- Generate evaluation metrics

**Note:** Training may take 10-30 minutes depending on your system and data size.

#### Step 6: Launch the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Quick Start (If Models Are Already Trained)

If you've cloned a repository that already includes trained models:

1. Clone the repository
2. Set up virtual environment and install dependencies (Steps 2-3 above)
3. Skip Step 5 (training) and go directly to Step 6 (launch the app)

### First-Time Setup

1. **Create an account**: Click on "Register" in the sidebar
   - Enter a valid email address
   - Create a strong password (8+ characters, mixed case, numbers, special characters)
   - Confirm your password
   - Accept the Terms & Conditions

2. **Create Admin User (Optional)**: 
   - After registering, you can create an admin account by:
     - Manually editing `app/credentials.json` and setting `"is_admin": true` for your email
     - Or use the Admin Panel's "Create Admin User" form (if you already have admin access)
   - Admin users have access to user management and system controls

3. **Login**: Use your credentials to access the dashboard

4. **Select a Bank**: Choose from available banks in the dropdown

5. **Set Forecast Horizon**: Adjust the slider to forecast 5-60 business days ahead

6. **Explore**: View statistics, charts, and download forecasts

## 📁 Project Structure

```
Stocks_Prediction/
├── app/
│   ├── models/          # Trained model files (.keras, .pkl, .json)
│   └── credentials.json  # User authentication data
├── data/                 # Bank stock CSV files
├── train_models.py       # Model training script
├── hybrid_model.py       # Hybrid ensemble forecasting
├── utils.py              # Utility functions
├── visualizations.py     # Plotly chart functions
├── streamlit_app.py      # Main Streamlit application
├── streamlit_auth.py     # Authentication helpers
├── install_dependencies.ps1
└── README.md
```

## 🎯 Usage Guide

### Viewing Statistics
- **Current Price**: Latest closing price with daily change
- **52-Week High/Low**: Price range over the past year
- **Volatility**: 30-day price volatility percentage

### Analyzing Models
- **Metrics Table**: Compare RMSE, MAE, and R² scores across models
- **Best Model Indicator**: Automatically highlights the best-performing model
- **Model Weights**: See how each model contributes to the hybrid forecast

### Viewing Forecasts
- **Historical Prices Tab**: View past price trends
- **Forecasts Tab**: See future predictions with summary statistics
- **Model Comparison Tab**: Compare actual vs predicted prices
- **Accuracy Metrics Tab**: Visual comparison of model performance

### Exporting Data
- **CSV Export**: Download forecast data as CSV
- **Excel Export**: Download forecast data as Excel spreadsheet
- **Auto-naming**: Files are automatically named with bank and date

## 🔧 Configuration

### Adjusting Forecast Horizon
- Use the slider in the main dashboard (5-60 business days)
- The forecast updates automatically when changed

### Model Training
- Models are trained with an 80/20 train/test split
- Training can be re-run anytime with `python train_models.py`
- Models are saved in `app/models/` directory

## 🛠️ Troubleshooting

### "No bank datasets found"
- Ensure CSV files are in the `data/` directory
- Files should follow the naming pattern: `{BANK}_stocks.csv`

### "Models not found"
- Run `python train_models.py` to train models
- Ensure models are saved in `app/models/` directory

### Authentication Issues
- Clear browser cache if login problems persist
- Check that `app/credentials.json` exists and is readable

### Excel Export Not Working
- Ensure `openpyxl` is installed: `pip install openpyxl`
- Re-run the installation script if needed

## 📊 Model Performance

The hybrid ensemble automatically weights models based on their inverse RMSE:
- **XGBoost**: Typically performs best (highest R², lowest RMSE)
- **LSTM**: Good for capturing long-term patterns
- **ARIMA**: Statistical baseline model
- **Hybrid**: Weighted combination for robust predictions

## 🔒 Security

- Passwords are hashed using Werkzeug's secure password hashing
- Credentials are stored locally in `app/credentials.json`
- No external authentication services required
- Session-based authentication with Streamlit

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For issues or questions, please check the troubleshooting section or review the code documentation.

---

**Built with**: Python, Streamlit, TensorFlow/Keras, XGBoost, Statsmodels, Plotly

