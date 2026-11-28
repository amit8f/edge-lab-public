"""AUTO1 candidate: engineered features + binary XGBoost classifier with stricter buy threshold."""
import numpy as np
import xgboost as xgb

# Original inputs
BASE_INPUTS = [
    "n_articles", "lm_neg", "lm_pos", "lm_net", "fb_pos", "fb_neg", "fb_max_neg", "fb_max_pos", "fb_neu",
    "head_len", "f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger",
    "f_fda", "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive", "time_bucket",
    "novelty", "r1", "r5", "r20", "vol20", "log_dv", "log_px", "spy_r1", "spy_r5", "mins_open",
    "r_open", "r_react", "rv30", "spy_open"
]

# Extra engineered features
def _engineered(df):
    # Gap between entry price and previous adjusted close
    gap = df["entry"] / df["prev_ac"] - 1.0
    # Gap expressed in units of recent volatility
    gap_vol = gap / df["vol20"].replace(0, np.nan)
    # Interaction of news reaction with prior trend
    react_trend = df["r_react"] - df["r5"]
    # Log of dollar volume (already present as log_dv, but keep raw dv20 for possible scaling)
    log_dv20 = np.log1p(df["dv20"])
    return np.column_stack([
        gap,
        gap_vol,
        react_trend,
        log_dv20,
    ])

def _build_features(df):
    base = df[BASE_INPUTS].to_numpy(float)
    extra = _engineered(df)
    return np.hstack([base, extra])

# Model parameters – classification
PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "random_state": 7,
    "n_jobs": 4,
    "eval_metric": "logloss",
}

# Threshold for buying: only scores > 0 are taken, so we subtract 0.6
BUY_THRESHOLD = 0.6

def fit(train):
    # Binary target: 1 if net return positive, else 0
    y = (train["net"] > 0).astype(int).to_numpy()
    X = _build_features(train)
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X, y)
    return model

def score(model, frame):
    X = _build_features(frame)
    # Predict probability of positive net; shift by threshold so that >0 indicates a buy
    prob = model.predict_proba(X)[:, 1]
    return prob - BUY_THRESHOLD
