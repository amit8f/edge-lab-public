"""AUTO3 start: NXM-ENS as frozen (mean of XGB, LGBM, LIN, RF and MLP news models, each over its own firing line;
fires at the training 90th percentile of that mean; 3 buys and 3 shorts a day)."""
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


def lgbm_learner(X, y):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=400, num_leaves=15, max_depth=4, learning_rate=0.05, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, min_child_samples=50, objective="huber",
                             alpha=1.0, random_state=20261024, deterministic=True, force_col_wise=True,
                             verbose=-1, n_jobs=4).fit(X, y)


def lin_learner(X, y):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=200)).fit(X, y)


def rf_learner(X, y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=200,
                                               max_samples=0.3, n_jobs=4, random_state=20261024)).fit(X, y)


def mlp_learner(X, y):
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3,
                                      max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                                      random_state=20261024)).fit(X, y)


class Ensemble:
    def __init__(self, members):
        self.members = members
        self.cut = float(np.quantile(np.abs(self.effect(members[0].real_X)), FIRE_Q))

    def effect(self, X):
        return np.mean([m.effect(X) / m.cut for m in self.members], axis=0)


def fit(train):
    return Ensemble([NewsModel(train, f) for f in (xgb_learner, lgbm_learner, lin_learner, rf_learner, mlp_learner)])


def score(model, frame):
    return model.effect(frame[INPUTS].to_numpy(float)) / model.cut
