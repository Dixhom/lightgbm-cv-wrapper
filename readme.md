# What is this?
This is a python class which wraps lightgbm.cv(). It has many features like:

- get oof
- predict using models in cv
- plot feature importance
- plot train logloss

It is compatible with binary and multi classification but not with regression.

# Usage

```python
lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 64,
    'min_data_in_leaf': 10,
    'max_depth': 5,
    'verbose': 0,
}

from lightgbmcv import LightGbmCv
lcv = LightGbmCv()
lcv.cv(X_train, y_train, lgbm_params, n_folds=5)

For detailed information, see examples.ipynb.
```

# Reference
- https://nigimitama.hatenablog.jp/entry/2021/01/05/205741
- https://blog.amedama.jp/entry/lightgbm-cv-model
- https://www.kaggle.com/code/kenmatsu4/using-trained-booster-from-lightgbm-cv-w-callback/notebook