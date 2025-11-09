# ============================================
# 📈 APPLE STOCK MARKET PRICE PREDICTION
# Modern Dark-Scientific Themed Dashboard (v3.0)
# ============================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- Streamlit Config ----
st.set_page_config(page_title="📈 Apple Stock Predictor", layout="wide", page_icon="📊")

# ---- Matplotlib Dark Research Theme ----
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0E1117',
    'axes.facecolor': '#0E1117',
    'axes.edgecolor': '#AAAAAA',
    'axes.labelcolor': '#FFFFFF',
    'xtick.color': '#BBBBBB',
    'ytick.color': '#BBBBBB',
    'grid.color': '#333333',
    'font.size': 12,
    'axes.titleweight': 'bold',
    'axes.titlepad': 10,
    'lines.linewidth': 1.8,
})

# ---- Title ----
st.title("📈 Apple Stock Market Price Prediction")
st.caption("Modern Dark-Scientific Theme | AI-Based Forecasting & Analysis")

# ---- File Upload ----
uploaded_file = st.sidebar.file_uploader("📂 Upload AAPL.csv", type=["csv"])
if not uploaded_file:
    st.warning("Please upload your dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
if "Date" not in df.columns:
    st.error("Dataset must include a 'Date' column.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").dropna().reset_index(drop=True)

# ---- Sidebar ----
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model", ["Linear Regression", "Decision Tree", "Random Forest", "SVR (RBF)"]
)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# ---- Feature Engineering ----
feature_cols = [c for c in ["Open", "High", "Low", "Volume"] if c in df.columns]
target_col = "Close"
X, y = df[feature_cols], df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# ---- Models ----
models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Decision Tree": Pipeline([("model", DecisionTreeRegressor(random_state=42))]),
    "Random Forest": Pipeline([("model", RandomForestRegressor(n_estimators=200, random_state=42))]),
    "SVR (RBF)": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
}

# ---- Train ----
chosen_model = models[model_choice]
chosen_model.fit(X_train, y_train)
pred = chosen_model.predict(X_test)

# ---- Evaluate ----
mse, rmse, mae, r2 = (
    mean_squared_error(y_test, pred),
    np.sqrt(mean_squared_error(y_test, pred)),
    mean_absolute_error(y_test, pred),
    r2_score(y_test, pred),
)

# ---- Metrics ----
st.markdown("### 📊 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")
col4.metric("MSE", f"{mse:.4f}")

# ---- Modern Predicted vs Actual Closing Price (Neon Style) ----
st.markdown("### 🎯 Predicted vs Actual Closing Price")
fig, ax = plt.subplots(figsize=(10, 5))

# Glow-style lines
ax.plot(y_test.values, color='#00E676', label='Actual', linewidth=2.2, alpha=0.9)
ax.plot(pred, color='#29B6F6', label='Predicted', linewidth=2.2, linestyle='--', alpha=0.9)
ax.fill_between(range(len(y_test)), y_test.values, pred, color='#1E88E5', alpha=0.08)

ax.set_title("Predicted vs Actual Closing Price", fontsize=14, color='white', pad=12)
ax.set_xlabel("Test Sample Index")
ax.set_ylabel("Close Price ($)")
ax.legend(facecolor='#0E1117', edgecolor='#444444')
ax.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig)

# ---- Error Distribution ----
st.markdown("### 📉 Error Distribution (Residuals)")
residuals = y_test.values - pred
fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.histplot(residuals, bins=25, color="#26C6DA", kde=True, ax=ax2)
ax2.set_title("Residual Distribution (Actual - Predicted)")
ax2.set_xlabel("Residual (Error)")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# ---- Volatility & Returns ----
st.markdown("### 📊 Price Volatility and Daily Returns")
df["Returns"] = df["Close"].pct_change()
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df["Date"], df["Returns"], color="#FFA726", linewidth=1.5)
ax3.axhline(0, color="#888", linestyle="--", linewidth=0.8)
ax3.set_title("Daily Returns Over Time")
ax3.set_ylabel("Return (%)")
st.pyplot(fig3)

# ---- Feature Importance ----
if model_choice in ["Decision Tree", "Random Forest"]:
    st.markdown("### 🌟 Feature Importance")
    importances = chosen_model.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=True)
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="mako", ax=ax4)
    ax4.set_title("Feature Importance (Model-Based)")
    st.pyplot(fig4)

# ---- CDF of Absolute Errors ----
st.markdown("### 📈 CDF of Absolute Errors (Model Reliability)")
errors = np.abs(y_test.values - pred)
sorted_err = np.sort(errors)
cdf = np.arange(len(errors)) / len(errors)
fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.plot(sorted_err, cdf, color='#00BCD4', linewidth=2)
ax5.axvline(np.percentile(errors, 50), linestyle="--", color="white", label=f"50% ≤ {np.percentile(errors, 50):.4f}")
ax5.axvline(np.percentile(errors, 90), linestyle="--", color="orange", label=f"90% ≤ {np.percentile(errors, 90):.4f}")
ax5.legend(facecolor='#111', edgecolor='#333')
ax5.set_xlabel("Absolute Error")
ax5.set_ylabel("Proportion ≤ Error")
ax5.set_title("CDF of Absolute Errors")
st.pyplot(fig5)

# ---- Correlation Heatmap ----
st.markdown("### 🔥 Correlation Heatmap (Feature Relationships)")
mask = np.triu(np.ones_like(df[["Open", "High", "Low", "Close", "Volume"]].corr(), dtype=bool))
fig6, ax6 = plt.subplots(figsize=(6, 5))
sns.heatmap(df[["Open", "High", "Low", "Close", "Volume"]].corr(), mask=mask, annot=True,
            cmap="coolwarm", center=0, cbar=True, linewidths=0.5, annot_kws={"size": 10})
ax6.set_title("Feature Correlation Matrix", fontsize=12)
st.pyplot(fig6)

# ---- Model Insight ----
st.markdown("### 💡 Automated Model Insight")
if r2 > 0.9:
    insight = "The model shows *excellent predictive accuracy* with minimal deviation — highly reliable."
elif r2 > 0.7:
    insight = "The model captures most price patterns well — slight deviations are within tolerance."
else:
    insight = "The model’s performance is moderate — try increasing data size or using ensemble models."
st.info(insight)

# ---- Next-Day Prediction ----
st.markdown("### 🔮 Predict Next-Day Closing Price")
cols = st.columns(len(feature_cols))
inputs = {}
for i, col in enumerate(feature_cols):
    inputs[col] = cols[i].number_input(f"{col}", value=float(df[col].iloc[-1]))

if st.button("🚀 Predict Next Day"):
    next_X = np.array([inputs[c] for c in feature_cols]).reshape(1, -1)
    next_pred = chosen_model.predict(next_X)[0]
    st.markdown(f"""
    <div style="border-radius:12px; border:2px solid #00E676; padding:20px; background:#111827; text-align:center;">
        <h3 style="color:#00E676;">Predicted NEXT-Day Close Price</h3>
        <h1 style="color:#29B6F6;">${next_pred:.2f}</h1>
        <p><b>Model:</b> {model_choice} | <b>R²:</b> {r2:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

# ---- Summary ----
st.markdown("---")
st.markdown(f"""
### ✅ Summary
- **Model Used:** {model_choice}  
- **R² Score:** {r2:.4f}  
- **MAE:** {mae:.4f} | **RMSE:** {rmse:.4f}  
- **Features Used:** {', '.join(feature_cols)}  
""")
