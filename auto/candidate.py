"""AUTO1 candidate: the only file the agent edits (research/prereg/AUTO1.md).

Start: an XGBoost regressor on NX3's 37 event inputs predicting each event's long net return after costs (column
"net"); buy when the prediction is above 0.
"""
import numpy as np
import xgboost as xgb

INPUTS = ["n_articles", "lm_neg", "lm_pos", "lm_net", "fb_pos", "fb_neg", "fb_max_neg", "fb_max_pos", "fb_neu",
          "head_len", "f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger", "f_fda",
          "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive", "time_bucket", "novelty", "r1", "r5",
          "r20", "vol20", "log_dv", "log_px", "spy_r1", "spy_r5", "mins_open", "r_open", "r_react", "rv30", "spy_open"]
PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 50, "objective": "reg:pseudohubererror", "tree_method": "hist", "random_state": 7, "n_jobs": 4}


def fit(train):
    y = np.clip(train["net"].to_numpy(float), -0.2, 0.2)
    return xgb.XGBRegressor(**PARAMS).fit(train[INPUTS].to_numpy(float), y)


def score(model, frame):
    return model.predict(frame[INPUTS].to_numpy(float))
