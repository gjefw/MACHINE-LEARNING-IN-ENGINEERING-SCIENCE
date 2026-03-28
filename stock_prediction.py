"""
S&P 500 Stock Price Prediction System
======================================
Models : XGBoost Regressor & Random Forest Regressor
Metric : Mean Squared Error (MSE)
Split  : Chronological (Train 2021-2024, Test 2025)
"""

import warnings
warnings.filterwarnings("ignore")

import io
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ============================================================
# Phase 1: Data Acquisition & Preprocessing
# ============================================================

def _download_via_yahoo_v8(ticker, start, end):
    """
    Download data from Yahoo Finance v8 chart API (JSON).
    This endpoint does not require cookie/crumb authentication.
    """
    import json

    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
    period1 = int(start_dt.timestamp())
    period2 = int(end_dt.timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = json.loads(resp.text)

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame({
        "Open":      quote["open"],
        "High":      quote["high"],
        "Low":       quote["low"],
        "Close":     quote["close"],
        "Adj Close": adj_close,
        "Volume":    quote["volume"],
    }, index=pd.to_datetime(timestamps, unit="s").normalize())
    df.index.name = "Date"
    return df


def download_data(ticker: str = "^GSPC",
                  start: str = "2021-01-01",
                  end: str = "2025-12-31") -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    print(f"[Phase 1] Downloading {ticker} data from {start} to {end} ...")

    # Try Yahoo Finance v8 chart API (no auth required)
    try:
        df = _download_via_yahoo_v8(ticker, start, end)
        print(f"  -> Downloaded {len(df)} rows (v8 chart API).\n")
    except Exception as e:
        print(f"  -> v8 API failed ({e}), trying yfinance library ...")
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"  -> Downloaded {len(df)} rows (yfinance).\n")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill isolated gaps, then drop remaining NaN rows."""
    before = len(df)
    df = df.ffill()
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"  -> Dropped {before - after} rows with missing values.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical features using ONLY past data.
    Target: next-day Close price (shift -1).
    """
    print("[Phase 1] Engineering features ...")

    # --- Returns ---
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)

    # --- Moving Averages ---
    df["SMA_5"]  = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()

    # --- Volatility ---
    df["Volatility_20"] = df["Return_1d"].rolling(window=20).std()

    # --- RSI (14-day) ---
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- Volume & Spread ---
    df["Volume_Change"]  = df["Volume"].pct_change(1)
    df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]

    # --- Target: next-day Close ---
    df["Target"] = df["Close"].shift(-1)

    # Replace inf/-inf with NaN, then drop all NaN rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feature_cols = [
        "Close", "Return_1d", "Return_5d",
        "SMA_5", "SMA_20", "EMA_12",
        "Volatility_20", "RSI_14",
        "Volume_Change", "High_Low_Spread",
    ]
    print(f"  -> Features: {feature_cols}")
    print(f"  -> Dataset size after feature engineering: {len(df)} rows.\n")
    return df, feature_cols


# ============================================================
# Phase 2: Strict Chronological Data Splitting
# ============================================================

def split_data(df: pd.DataFrame, feature_cols: list,
               train_end: str = "2024-12-31",
               test_start: str = "2025-01-01"):
    """
    Split data by date index.  NO random shuffling.
    """
    print("[Phase 2] Splitting data chronologically ...")

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    train = df.loc[:train_end]
    test  = df.loc[test_start:]

    # Assertion: no overlap
    assert train.index.max() < test.index.min(), \
        "DATA LEAK: train dates overlap with test dates!"

    X_train, y_train = train[feature_cols], train["Target"]
    X_test,  y_test  = test[feature_cols],  test["Target"]

    print(f"  -> Train: {X_train.index.min().date()} - {X_train.index.max().date()}  ({len(X_train)} rows)")
    print(f"  -> Test : {X_test.index.min().date()} - {X_test.index.max().date()}  ({len(X_test)} rows)")
    print(f"  -> Assertion passed: max(train) < min(test) [OK]\n")

    return X_train, y_train, X_test, y_test


def scale_features(X_train, X_test):
    """Fit scaler on Train only, transform both."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    print("[Phase 2] Feature scaling applied (fitted on Train only). [OK]\n")
    return X_train_scaled, X_test_scaled


# ============================================================
# Phase 3: Model Training
# ============================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train an XGBoost Regressor with early stopping."""
    print("[Phase 3] Training XGBoost Regressor ...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    preds = model.predict(X_test)
    print(f"  -> Best iteration: {model.best_iteration}\n")
    return model, preds


def train_random_forest(X_train, y_train, X_test):
    """Train a Random Forest Regressor."""
    print("[Phase 3] Training Random Forest Regressor ...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("  -> Training complete.\n")
    return model, preds


# ============================================================
# Phase 4: Evaluation & Visualization
# ============================================================

def evaluate(y_test, pred_xgb, pred_rf):
    """Calculate and display MSE for both models."""
    mse_xgb = mean_squared_error(y_test, pred_xgb)
    mse_rf  = mean_squared_error(y_test, pred_rf)

    print("=" * 50)
    print("        MODEL EVALUATION - MSE COMPARISON")
    print("=" * 50)
    print(f"  {'Model':<25} {'MSE':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'XGBoost':<25} {mse_xgb:>15,.2f}")
    print(f"  {'Random Forest':<25} {mse_rf:>15,.2f}")
    print("=" * 50)

    winner = "XGBoost" if mse_xgb < mse_rf else "Random Forest"
    print(f"\n  * Best model: {winner}\n")
    return mse_xgb, mse_rf, winner


def plot_predictions(y_test, pred_xgb, pred_rf, save_path="prediction_results.png"):
    """Plot Actual vs. Predicted prices for the test period."""
    print("[Phase 4] Generating Actual vs. Predicted chart ...")
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(y_test.index, y_test.values, color="black",
            linewidth=2, label="Actual", zorder=3)
    ax.plot(y_test.index, pred_xgb, color="#1f77b4",
            linewidth=1.5, linestyle="--", label="XGBoost Predicted", alpha=0.85)
    ax.plot(y_test.index, pred_rf, color="#ff7f0e",
            linewidth=1.5, linestyle="--", label="Random Forest Predicted", alpha=0.85)

    ax.set_title("S&P 500 - Actual vs. Predicted Close Price (2025 Test Period)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price (USD)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved to {save_path}\n")
    plt.close(fig)


def plot_feature_importance(model, feature_cols, model_name, save_path="feature_importance.png"):
    """Horizontal bar chart of top-10 feature importances."""
    print(f"[Phase 4] Generating feature importance chart ({model_name}) ...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    fig, ax = plt.subplots(figsize=(10, 5))
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importances[::-1],
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=11)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top-10 Feature Importances - {model_name}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  -> Saved to {save_path}\n")
    plt.close(fig)


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  S&P 500 STOCK PRICE PREDICTION SYSTEM")
    print("  XGBoost | Random Forest | Chronological Split")
    print("=" * 60 + "\n")

    # Phase 1
    df = download_data()
    df = handle_missing_values(df)
    df, feature_cols = engineer_features(df)

    # Phase 2
    X_train, y_train, X_test, y_test = split_data(df, feature_cols)
    X_train_s, X_test_s = scale_features(X_train, X_test)

    # Phase 3
    model_xgb, pred_xgb = train_xgboost(X_train_s, y_train, X_test_s, y_test)
    model_rf,  pred_rf  = train_random_forest(X_train_s, y_train, X_test_s)

    # Phase 4
    mse_xgb, mse_rf, winner = evaluate(y_test, pred_xgb, pred_rf)
    plot_predictions(y_test, pred_xgb, pred_rf)
    plot_feature_importance(
        model_xgb if winner == "XGBoost" else model_rf,
        feature_cols,
        winner,
    )

    print("[DONE] Pipeline complete. All outputs saved.\n")


if __name__ == "__main__":
    main()
