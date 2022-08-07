import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns

class LightGbmCv:
    def __init__(self):
        pass
    def cv(self, X_train, y_train, lgbm_params, n_folds=5):
        assert np.unique(y_train).size == y_train.max() + 1
        assert y_train.min() == 0
        
        # get classification type
        self.max_class_num = y_train.max() + 1
        if self.max_class_num < 2:
            raise ValueError('max number of classes was less than 2')
        elif self.max_class_num == 2:
            self.classification_type = 'binary'
        else:
            self.classification_type = 'multi'

        # one hot representation of y_train
        if self.classification_type == 'binary':    
            self.y_train_ohe = y_train
        elif self.classification_type == 'multi':
            self.y_train_ohe = np.identity(self.max_class_num)[y_train]
        else:
            raise ValueError('classfication type is invalid: ' + self.classification_type)

        # turn the data into LightGBM representation 
        lgb_train = lgb.Dataset(X_train, y_train)
        
        # get folds
        self.folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

        # Fitting
        # https://nigimitama.hatenablog.jp/entry/2021/01/05/205741
        cv_result = lgb.cv(params=lgbm_params,
                           train_set=lgb_train,
                           folds=self.folds,
                           num_boost_round=1000,
                           verbose_eval = 10,
                           early_stopping_rounds=10,
                           return_cvbooster=True,
                           seed=0
                           )
        
        # this has logloss in it
        self.cv_result = cv_result
        # this has feature importances and so on
        self.cvbooster = cv_result['cvbooster']
        # best iteration
        self.best_iteration = self.cvbooster.best_iteration
        # models
        self.boosters = self.cvbooster.boosters
        
    def get_oof(self, X_train, y_train, binary_threshold=0.5):
        # get the same folds as lgb.cv()
        fold_iter = self.folds.split(X_train, y_train)
        # empty numpy array to put in values
        oof_preds_proba = np.zeros_like(self.y_train_ohe, dtype=np.float32)
        oof_preds_label = np.zeros(self.y_train_ohe.shape[0], dtype=np.float32)
        # iterate folds and boosters to retrieve oof
        for n_fold, ((trn_idx, val_idx), booster) in enumerate(zip(fold_iter, self.boosters)):
            valid = X_train[val_idx]
            y_pred_proba = booster.predict(valid, num_iteration=self.best_iteration)
            oof_preds_proba[val_idx] = y_pred_proba
            if self.classification_type == 'binary':
                oof_preds_label[val_idx] = (y_pred_proba > binary_threshold).astype(int)
            else:
                oof_preds_label[val_idx] = np.argmax(y_pred_proba, axis=1)
                
        return oof_preds_proba, oof_preds_label 
    
    def predict(self, X_test, binary_threshold=0.5):
        # binary classification
        if self.classification_type == 'binary':
            # prepare container
            pred_proba = np.zeros(len(X_test), dtype=np.float32)
            # iterate boosters and predict
            for booster in self.boosters:
                pred_proba += booster.predict(X_test, num_iteration=self.best_iteration)
            # get mean
            pred_proba /= len(self.boosters)
            # get label
            pred_label = (pred_proba > binary_threshold).astype(int)
        # multi classification
        else:
            # prepare container
            pred_proba = np.zeros((len(X_test), self.max_class_num), dtype=np.float32)
            # iterate boosters and predict
            for booster in self.boosters:
                pred_proba += booster.predict(X_test, num_iteration=self.best_iteration)
            # get mean
            pred_proba /= len(self.boosters)
            # get label
            pred_label = np.argmax(pred_proba, axis=1)
            
        return pred_proba, pred_label
    
    def plot_feature_importance(self, plot_top_n=20):
        # when plot_top_n < 0, the last plot_top features will be shown
        assert plot_top_n != 0
        
        # feature importance
        raw_importances = self.cvbooster.feature_importance(importance_type='gain')
        # feature names
        feature_name = self.cvbooster.boosters[0].feature_name()
        # dataframe
        importance_df = pd.DataFrame(data=raw_importances,
                                     columns=feature_name)

        # sort features by mean
        sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
        sorted_importance_df = importance_df.loc[:, sorted_indices]
        # plot top N features
        if plot_top_n > 0:
            plot_cols = sorted_importance_df.columns[:plot_top_n]
        else:
            plot_cols = sorted_importance_df.columns[plot_top_n:]
        
        _, ax = plt.subplots(figsize=(8, 8))
        ax.grid()
        ax.set_xscale('log')
        ax.set_ylabel('Feature')
        ax.set_xlabel('Importance')
        sns.boxplot(data=sorted_importance_df[plot_cols],
                    orient='h',
                    ax=ax)
        plt.show()
        
    def plot_train_logloss(self):
        # binary classification
        if self.classification_type == 'binary':
            loss = pd.DataFrame(dict(mean=self.cv_result['binary_logloss-mean'], std=self.cv_result['binary_logloss-stdv']))
        # multi classification
        else:
            loss = pd.DataFrame(dict(mean=self.cv_result['multi_logloss-mean'], std=self.cv_result['multi_logloss-stdv']))
        loss.plot(yerr='std')