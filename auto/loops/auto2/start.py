"""AUTO2 start: NX3-HI as frozen (NX3's XGBoost news model; trade at confidence |e|/cut >= 1.5, 3 buys and 3
shorts a day). The harness trades score >= 1 / <= -1, so score = e / (1.5 * cut)."""
import numpy as np

INPUTS = ["n_articles", "lm_neg", "lm_pos", "lm_net", "fb_pos", "fb_neg", "fb_max_neg", "fb_max_pos", "fb_neu",
          "head_len", "f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger", "f_fda",
          "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive", "time_bucket", "novelty", "r1", "r5",
          "r20", "vol20", "log_dv", "log_px", "spy_r1", "spy_r5", "mins_open", "r_open", "r_react", "rv30", "spy_open"]
NEWS = INPUTS[:23]                       # set to 0 for the "no news" twin of an event
BASE = ["r1", "r5", "r20", "vol20", "log_dv", "log_px", "spy_r1", "spy_r5", "mins_open", "r_open", "r_react", "rv30",
        "spy_open"]
FIRE_Q, Z_CLIP, HOLD_SCALE = 0.90, 6.0, 1.5 ** 0.5


def zlabel(f):
    v = f["vol20"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = f["mret"].to_numpy(float) / (v * HOLD_SCALE)
    z[~(v > 0)] = np.nan
    return np.clip(z, -Z_CLIP, Z_CLIP)


def no_news(X):
    X0 = X.copy()
    for i, k in enumerate(INPUTS):
        if k in NEWS:
            X0[:, i] = 0.0
    return X0


class NewsModel:
    """NX3's design: a baseline fitted on placebo (no-news) events, a news model on real events' label minus the
    baseline; effect = news(x) - news(x with the news inputs at 0); fires at the training 90th percentile."""

    def __init__(self, train, learner):
        z = zlabel(train)
        t = train.assign(z=z)[np.isfinite(z)]
        pl, re = t[t["kind"] == "placebo"], t[t["kind"] == "real"]
        self.base = learner(pl[BASE].to_numpy(float), pl["z"].to_numpy(float))
        X = re[INPUTS].to_numpy(float)
        self.news = learner(X, re["z"].to_numpy(float) - self.base.predict(re[BASE].to_numpy(float)))
        self.cut = float(np.quantile(np.abs(self.effect(X)), FIRE_Q))
        self.real_X = X

    def effect(self, X):
        return self.news.predict(X) - self.news.predict(no_news(X))


def xgb_learner(X, y):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=50, objective="reg:pseudohubererror", huber_slope=1.0, tree_method="hist",
                            random_state=20261024, n_jobs=4).fit(X, y)


BAR = 1.5


def fit(train):
    return NewsModel(train, xgb_learner)


def score(model, frame):
    return model.effect(frame[INPUTS].to_numpy(float)) / (BAR * model.cut)
