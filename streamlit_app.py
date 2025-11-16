from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

from hybrid_model import ForecastOutputs, generate_forecasts
from streamlit_auth import (
    authenticate_user,
    delete_user,
    ensure_credentials_file,
    get_all_users,
    is_admin_user,
    register_user,
)
from utils import discover_datasets, load_bank_dataset, load_metrics
from visualizations import (
    plot_future_forecast,
    plot_historical_prices,
    plot_model_backtests,
    plot_model_metrics,
)

st.set_page_config(
    page_title="Kenyan Bank Stock Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_credentials_file()

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "email" not in st.session_state:
    st.session_state.email = None
if "selected_bank" not in st.session_state:
    st.session_state.selected_bank = None
if "horizon" not in st.session_state:
    st.session_state.horizon = 15


def inject_custom_css() -> None:
    """Inject custom CSS for enhanced UI."""
    st.markdown(
        """
        <style>
            /* Main container */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Header styling */
            h1 {
                color: #1f77b4;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 0.5rem;
            }
            
            /* Metric cards */
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                padding-top: 3rem;
            }
            
            /* Button styling */
            .stButton > button {
                width: 100%;
                border-radius: 5px;
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 600;
                transition: all 0.3s;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* Input styling */
            .stTextInput > div > div > input {
                border-radius: 5px;
            }
            
            /* Tabs styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.75rem 1.5rem;
                border-radius: 0.5rem 0.5rem 0 0;
                background-color: #f0f2f6;
                font-weight: 500;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            /* Success/Error messages */
            .stSuccess {
                border-radius: 5px;
            }
            
            .stError {
                border-radius: 5px;
            }
            
            /* Info boxes */
            .info-box {
                background-color: #e3f2fd;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
                margin: 1rem 0;
                color: #1a1a1a;
            }
            
            .info-box h3 {
                color: #1976d2;
                margin-top: 0;
                margin-bottom: 1rem;
            }
            
            .info-box ul {
                color: #333;
                margin: 0;
                padding-left: 1.5rem;
            }
            
            .info-box li {
                margin-bottom: 0.5rem;
                line-height: 1.6;
            }
            
            .info-box strong {
                color: #1565c0;
            }
            
            /* Password strength indicator */
            .password-strength {
                height: 4px;
                border-radius: 2px;
                margin-top: 0.5rem;
            }
            
            .strength-weak { background-color: #f44336; }
            .strength-medium { background-color: #ff9800; }
            .strength-strong { background-color: #4caf50; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_password_strength(password: str) -> tuple[str, float]:
    """Check password strength and return strength level and score."""
    score = 0
    feedback = []
    
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("At least 8 characters")
    
    if re.search(r"[a-z]", password) and re.search(r"[A-Z]", password):
        score += 1
    else:
        feedback.append("Mix of uppercase and lowercase")
    
    if re.search(r"\d", password):
        score += 1
    else:
        feedback.append("At least one number")
    
    if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        score += 1
    else:
        feedback.append("At least one special character")
    
    if len(password) >= 12:
        score += 0.5
    
    if score <= 1.5:
        return "weak", score / 4
    elif score <= 2.5:
        return "medium", score / 4
    else:
        return "strong", score / 4


def display_password_strength(password: str) -> None:
    """Display password strength indicator."""
    if not password:
        return
    
    strength, score = check_password_strength(password)
    width = int(score * 100)
    
    color_map = {
        "weak": "#f44336",
        "medium": "#ff9800",
        "strong": "#4caf50",
    }
    
    st.markdown(
        f"""
        <div class="password-strength" style="background-color: {color_map[strength]}; width: {width}%;"></div>
        <small style="color: {color_map[strength]}; font-weight: 600;">Password Strength: {strength.upper()}</small>
        """,
        unsafe_allow_html=True,
    )


def calculate_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key statistics for the dataset."""
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
    change = latest_close - prev_close
    change_pct = (change / prev_close * 100) if prev_close != 0 else 0
    
    return {
        "current_price": latest_close,
        "change": change,
        "change_pct": change_pct,
        "high_52w": df["Close"].tail(252).max() if len(df) >= 252 else df["Close"].max(),
        "low_52w": df["Close"].tail(252).min() if len(df) >= 252 else df["Close"].min(),
        "avg_volume": df["Volume"].tail(30).mean() if "Volume" in df.columns else 0,
        "volatility": df["Close"].pct_change().tail(30).std() * 100 if len(df) > 1 else 0,
    }


def display_statistics_cards(stats: Dict[str, float], bank: str) -> None:
    """Display statistics in card format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"KES {stats['current_price']:.2f}",
            delta=f"{stats['change']:+.2f} ({stats['change_pct']:+.2f}%)",
            delta_color="normal" if stats['change'] >= 0 else "inverse",
        )
    
    with col2:
        st.metric(
            label="📊 52W High",
            value=f"KES {stats['high_52w']:.2f}",
        )
    
    with col3:
        st.metric(
            label="📉 52W Low",
            value=f"KES {stats['low_52w']:.2f}",
        )
    
    with col4:
        st.metric(
            label="📈 Volatility (30D)",
            value=f"{stats['volatility']:.2f}%",
        )


def calculate_forecast_statistics(future_df: pd.DataFrame, current_price: float) -> Dict[str, float]:
    """Calculate statistics from forecast data."""
    hybrid_forecast = future_df["Hybrid"]
    
    forecast_high = hybrid_forecast.max()
    forecast_low = hybrid_forecast.min()
    forecast_avg = hybrid_forecast.mean()
    forecast_start = hybrid_forecast.iloc[0]
    forecast_end = hybrid_forecast.iloc[-1]
    
    # Calculate expected change
    expected_change = forecast_end - current_price
    expected_change_pct = (expected_change / current_price * 100) if current_price != 0 else 0
    
    # Calculate forecast range
    forecast_range = forecast_high - forecast_low
    forecast_range_pct = (forecast_range / forecast_avg * 100) if forecast_avg != 0 else 0
    
    # Calculate forecast volatility (std of forecast values)
    forecast_volatility = hybrid_forecast.pct_change().std() * 100 if len(hybrid_forecast) > 1 else 0
    
    return {
        "forecast_high": forecast_high,
        "forecast_low": forecast_low,
        "forecast_avg": forecast_avg,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "expected_change": expected_change,
        "expected_change_pct": expected_change_pct,
        "forecast_range": forecast_range,
        "forecast_range_pct": forecast_range_pct,
        "forecast_volatility": forecast_volatility,
    }


def display_forecast_statistics_cards(stats: Dict[str, float], bank: str) -> None:
    """Display forecast-based statistics in card format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Forecast High",
            value=f"KES {stats['forecast_high']:.2f}",
            delta=f"Range: {stats['forecast_range']:.2f}",
        )
    
    with col2:
        st.metric(
            label="📉 Forecast Low",
            value=f"KES {stats['forecast_low']:.2f}",
        )
    
    with col3:
        st.metric(
            label="📊 Forecast Average",
            value=f"KES {stats['forecast_avg']:.2f}",
        )
    
    with col4:
        st.metric(
            label="🎯 Expected Change",
            value=f"{stats['expected_change_pct']:+.2f}%",
            delta=f"{stats['expected_change']:+.2f}",
            delta_color="normal" if stats['expected_change'] >= 0 else "inverse",
        )
    
    # Additional forecast metrics
    col5, col6 = st.columns(2)
    with col5:
        st.metric(
            label="📏 Forecast Range",
            value=f"{stats['forecast_range_pct']:.2f}%",
        )
    
    with col6:
        st.metric(
            label="📈 Forecast Volatility",
            value=f"{stats['forecast_volatility']:.2f}%",
        )


@st.cache_data(show_spinner=False)
def get_available_banks() -> Dict[str, str]:
    return discover_datasets()


@st.cache_data(show_spinner=False)
def load_dataset(bank: str) -> pd.DataFrame:
    return load_bank_dataset(bank)


@st.cache_data(show_spinner=False)
def load_bank_metrics(bank: str) -> Dict[str, Dict[str, float]]:
    return load_metrics(bank)


@st.cache_data(show_spinner=True)
def generate_cached_forecast(bank: str, horizon: int) -> ForecastOutputs:
    return generate_forecasts(bank, steps=horizon)


def display_metrics_table(metrics: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> None:
    """Display metrics in an enhanced table format."""
    st.subheader("📊 Model Performance Metrics")
    
    metric_df = pd.DataFrame(metrics).T
    if "weights" in metric_df.columns:
        metric_df = metric_df.drop(columns=["weights"])
    
    weight_series = pd.Series(weights, name="Weight")
    metric_df = metric_df.join(weight_series, how="left")
    
    # Style the dataframe
    styled_df = metric_df.style.format(
        {
            "RMSE": "{:.3f}",
            "MAE": "{:.3f}",
            "R2": "{:.3f}",
            "Weight": "{:.1%}",
        }
    ).background_gradient(subset=["R2"], cmap="Greens").background_gradient(
        subset=["RMSE", "MAE"], cmap="Reds_r"
    )
    
    st.dataframe(styled_df, width='stretch', height=200)
    
    # Best model indicator
    if "R2" in metric_df.columns:
        best_model = metric_df["R2"].idxmax()
        st.info(f"🏆 **Best Performing Model:** {best_model} (R² = {metric_df.loc[best_model, 'R2']:.3f})")


def display_charts(bank: str, dataset: pd.DataFrame, forecasts: ForecastOutputs, metrics: Dict[str, Dict[str, float]]) -> None:
    """Display enhanced visualizations."""
    st.subheader("📈 Visualizations")
    
    # Tabs for different chart views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical Prices", "🔮 Forecasts", "📉 Model Comparison", "🎯 Accuracy Metrics"])
    
    with tab1:
        hist_fig = plot_historical_prices(dataset, bank)
        st.plotly_chart(hist_fig, width='stretch')
    
    with tab2:
        future_fig = plot_future_forecast(forecasts.future, bank)
        st.plotly_chart(future_fig, width='stretch')
        
        # Forecast summary
        col1, col2, col3 = st.columns(3)
        future_df = forecasts.future
        with col1:
            st.metric("📈 Forecast High", f"KES {future_df['Hybrid'].max():.2f}")
        with col2:
            st.metric("📉 Forecast Low", f"KES {future_df['Hybrid'].min():.2f}")
        with col3:
            st.metric("📊 Forecast Avg", f"KES {future_df['Hybrid'].mean():.2f}")
    
    with tab3:
        backtest_frames = dict(forecasts.historical)
        backtest_frames["Hybrid"] = forecasts.hybrid_historical
        backtest_fig = plot_model_backtests(backtest_frames, bank)
        st.plotly_chart(backtest_fig, width='stretch')
    
    with tab4:
        metrics_fig = plot_model_metrics(metrics)
        st.plotly_chart(metrics_fig, width='stretch')


def display_forecast_table(future_df: pd.DataFrame, bank: str) -> None:
    """Display forecast table with export options."""
    st.subheader("📋 Detailed Forecast Table")
    
    # Format the dataframe
    styled_df = future_df.style.format("{:.2f}").background_gradient(
        subset=["Hybrid"], cmap="YlOrRd"
    )
    
    st.dataframe(styled_df, width='stretch', height=400)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_buffer = io.StringIO()
        future_df.to_csv(csv_buffer)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{bank}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width='stretch',
        )
    
    with col2:
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                future_df.to_excel(writer, sheet_name='Forecast')
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{bank}_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch',
            )
        except ImportError:
            st.info("💡 Install openpyxl for Excel export: `pip install openpyxl`")
        except Exception as e:
            st.warning(f"⚠️ Excel export unavailable: {str(e)}")
    
    with col3:
        if st.button("🔄 Refresh Forecast", width='stretch'):
            st.cache_data.clear()
            st.rerun()


def login_section() -> None:
    """Enhanced login/registration section."""
    st.sidebar.markdown("## 🔐 Account")
    
    if st.session_state.get("authenticated"):
        st.sidebar.success(f"✅ Logged in as\n**{st.session_state['email']}**")
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🚪 Log Out", width='stretch', type="primary"):
            st.session_state.clear()
            st.rerun()
        
        # User info
        with st.sidebar.expander("ℹ️ Account Info"):
            st.write(f"**Email:** {st.session_state['email']}")
            is_admin = is_admin_user(st.session_state['email'])
            st.write(f"**Role:** {'👑 Admin' if is_admin else '👤 User'}")
            st.write(f"**Session:** Active")
        return
    
    # Login/Register tabs
    tabs = st.sidebar.tabs(["🔑 Login", "📝 Register"])
    
    with tabs[0]:
        st.markdown("### Welcome Back!")
        login_email = st.text_input("📧 Email Address", key="login_email", placeholder="your.email@example.com")
        login_password = st.text_input("🔒 Password", type="password", key="login_password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            remember_me = st.checkbox("Remember me", key="remember")
        
        if st.button("🚀 Sign In", key="login_button", width='stretch', type="primary"):
            if not login_email or not login_password:
                st.error("⚠️ Please fill in all fields.")
            elif authenticate_user(login_email.strip().lower(), login_password):
                st.session_state.authenticated = True
                st.session_state.email = login_email.strip().lower()
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid email or password. Please try again.")
    
    with tabs[1]:
        st.markdown("### Create New Account")
        reg_email = st.text_input("📧 Email Address", key="register_email", placeholder="your.email@example.com")
        reg_password = st.text_input("🔒 Password", type="password", key="register_password", placeholder="Create a strong password")
        
        if reg_password:
            display_password_strength(reg_password)
        
        reg_confirm = st.text_input("🔒 Confirm Password", type="password", key="register_confirm", placeholder="Re-enter your password")
        
        # Terms and conditions
        terms = st.checkbox("I agree to the Terms & Conditions", key="terms")
        
        if st.button("✨ Create Account", key="register_button", width='stretch', type="primary"):
            if not reg_email or not reg_password or not reg_confirm:
                st.error("⚠️ Please fill in all fields.")
            elif not terms:
                st.error("⚠️ Please accept the Terms & Conditions.")
            elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', reg_email):
                st.error("⚠️ Please enter a valid email address.")
            elif reg_password != reg_confirm:
                st.error("❌ Passwords do not match.")
            elif len(reg_password) < 8:
                st.error("⚠️ Password must be at least 8 characters long.")
            else:
                success, message = register_user(reg_email.strip().lower(), reg_password)
                if success:
                    st.success(f"✅ {message} You can now log in.")
                else:
                    st.error(f"❌ {message}")


def display_admin_panel() -> None:
    """Display admin panel with user management and system controls."""
    st.title("👑 Admin Panel")
    st.markdown("---")
    
    # User Management Section
    st.subheader("👥 User Management")
    
    users = get_all_users()
    if users:
        users_df = pd.DataFrame(users)
        st.dataframe(users_df, width='stretch', height=300)
        
        # Delete user
        st.markdown("### Delete User")
        col1, col2 = st.columns([3, 1])
        with col1:
            user_to_delete = st.selectbox(
                "Select user to delete",
                options=[u["email"] for u in users if u["email"] != st.session_state['email']],
                key="delete_user_select",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", width='stretch', type="secondary"):
                if user_to_delete:
                    success, message = delete_user(user_to_delete)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
    else:
        st.info("No users found.")
    
    st.markdown("---")
    
    # System Management Section
    st.subheader("⚙️ System Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Clear Cache", width='stretch', type="primary"):
            st.cache_data.clear()
            st.success("✅ Cache cleared successfully!")
    
    with col2:
        if st.button("📊 View System Stats", width='stretch'):
            from pathlib import Path
            model_dir = Path("app/models")
            data_dir = Path("data")
            
            model_count = len(list(model_dir.glob("*.pkl"))) + len(list(model_dir.glob("*.keras")))
            data_files = len(list(data_dir.glob("*.csv")))
            
            st.info(f"""
            **System Statistics:**
            - Trained Models: {model_count}
            - Data Files: {data_files}
            - Total Users: {len(users)}
            - Admin Users: {sum(1 for u in users if u.get('is_admin', False))}
            """)
    
    with col3:
        if st.button("🔄 Retrain Models", width='stretch', type="secondary"):
            st.warning("⚠️ This will retrain all models. This may take a while.")
            if st.button("Confirm Retrain", key="confirm_retrain"):
                import subprocess
                with st.spinner("Training models..."):
                    result = subprocess.run(["python", "train_models.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ Models retrained successfully!")
                        st.cache_data.clear()
                    else:
                        st.error(f"❌ Training failed: {result.stderr}")
    
    st.markdown("---")
    
    # Create Admin User
    st.subheader("➕ Create Admin User")
    with st.form("create_admin_form"):
        admin_email = st.text_input("Email Address", key="admin_email")
        admin_password = st.text_input("Password", type="password", key="admin_password")
        admin_confirm = st.text_input("Confirm Password", type="password", key="admin_confirm")
        
        if st.form_submit_button("✨ Create Admin", width='stretch', type="primary"):
            if admin_password != admin_confirm:
                st.error("❌ Passwords do not match.")
            elif len(admin_password) < 8:
                st.error("⚠️ Password must be at least 8 characters.")
            else:
                success, message = register_user(admin_email.strip().lower(), admin_password, is_admin=True)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")


def main() -> None:
    """Main application function."""
    inject_custom_css()
    login_section()
    
    if not st.session_state.get("authenticated"):
        # Landing page for non-authenticated users
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 0;">
                <h1 style="color: #1f77b4; font-size: 3rem;">📈 Kenyan Bank Stock Forecast</h1>
                <p style="font-size: 1.5rem; color: #666;">Advanced ML-Powered Stock Price Prediction</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div class="info-box">
                    <h3>🎯 Features</h3>
                    <ul>
                        <li>🤖 <strong>Hybrid ML Models:</strong> LSTM, ARIMA, and XGBoost</li>
                        <li>📊 <strong>Interactive Visualizations:</strong> Real-time charts and forecasts</li>
                        <li>📈 <strong>Accurate Predictions:</strong> Advanced ensemble methods</li>
                        <li>💾 <strong>Export Options:</strong> CSV and Excel downloads</li>
                        <li>🔒 <strong>Secure Access:</strong> User authentication system</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.info("👆 Please log in or create an account in the sidebar to access the dashboard.")
        return
    
    # Check if user is admin and show admin option
    is_admin = is_admin_user(st.session_state['email'])
    
    # Main dashboard with tabs
    if is_admin:
        tab1, tab2 = st.tabs(["📊 Dashboard", "👑 Admin Panel"])
    else:
        tab1 = st.container()
        tab2 = None
    
    with tab1:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h1>📈 Kenyan Bank Stock Forecast Dashboard</h1>
                <p style="color: #666; font-size: 0.9rem;">Welcome, {st.session_state['email']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        datasets = get_available_banks()
        banks = sorted(datasets.keys())
        
        if not banks:
            st.warning("⚠️ No bank datasets found. Please add CSV files to the data directory.")
            return
        
        # Controls section
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            default_bank = banks[0]
            selected_bank = st.selectbox(
                "🏦 Select a Bank",
                options=banks,
                index=banks.index(st.session_state.get("selected_bank", default_bank))
                if st.session_state.get("selected_bank", default_bank) in banks
                else 0,
                help="Choose a bank to analyze and forecast",
            )
            st.session_state.selected_bank = selected_bank
        
        with col2:
            horizon = st.slider(
                "📅 Forecast Horizon (Business Days)",
                min_value=5,
                max_value=60,
                value=st.session_state.get("horizon", 15),
                help="Number of business days to forecast ahead",
            )
            st.session_state.horizon = horizon
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Data", width='stretch'):
                st.cache_data.clear()
                st.rerun()
        
        # Load data
        try:
            with st.spinner("🔄 Loading data and generating forecasts..."):
                dataset = load_dataset(selected_bank)
                metrics = load_bank_metrics(selected_bank)
                forecasts = generate_cached_forecast(selected_bank, horizon)
            
            # Display statistics
            st.markdown("---")
            st.subheader("📊 Historical Statistics")
            stats = calculate_statistics(dataset)
            display_statistics_cards(stats, selected_bank)
            
            # Display forecast-based statistics that change with horizon
            st.markdown("---")
            st.subheader(f"🔮 Forecast Statistics (Next {horizon} Days)")
            forecast_stats = calculate_forecast_statistics(forecasts.future, dataset["Close"].iloc[-1])
            display_forecast_statistics_cards(forecast_stats, selected_bank)
            
            st.markdown("---")
            
            # Display metrics
            display_metrics_table(metrics, forecasts.weights)
            
            st.markdown("---")
            
            # Display charts
            display_charts(selected_bank, dataset, forecasts, metrics)
            
            st.markdown("---")
            
            # Display forecast table
            display_forecast_table(forecasts.future, selected_bank)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("💡 Please ensure models are trained by running `python train_models.py`")
    
    # Admin Panel Tab
    if is_admin and tab2:
        with tab2:
            display_admin_panel()


if __name__ == "__main__":
    main()
