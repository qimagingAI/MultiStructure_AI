'''
Init, fit, and evaluate the xgboost model with a single set of params and single split
'''
import itertools
import pickle
import time
import imblearn
import numpy as np
import optuna
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb

from statistics import mean
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, RepeatedStratifiedKFold, \
    LeaveOneGroupOut
from xgboost import XGBClassifier


def normalize_df(df):
    '''
    default [0,1] normalization
    '''
    X2 = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X2)
    X2_scaled = min_max_scaler.transform(X2)
    df = pd.DataFrame(X2_scaled, columns=df.columns)
    return df


def get_params_list(params):
    hyperparams_list = []
    print(params.items())
    keys = params.keys()
    ll = list(itertools.product(*[value for _, value in params.items()]))
    for param_set in ll:
        hyperparams_dummy = {}
        for i, key in enumerate(keys):
            hyperparams_dummy[key] = param_set[i]
        hyperparams_list.append(hyperparams_dummy)
    return hyperparams_list


def eval_xgboost_kfold_grid_clean(dataframes_list, preds, fis, params, set_id=1, label="", normalize=False, folds=10,
                                  id_col="", save_model=True, save_model_pth='./', seed=0, under_sample=False,
                                  split_col=None):
    print("Starting internal xgboost k-fold validation...")

    # parse out the dataframes from the dataframes_list object 
    if len(dataframes_list) != 2:
        raise ValueError(f"Length of dataframes_list must be 2, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y = dataframes_list[0], dataframes_list[1].astype('int')
    X_orig = X.copy()
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index
    if normalize:
        X = normalize_df(X)

    print("Feature Shape Internal: ", X.shape)
    print("Label Shape Internal", y.shape)

    if folds > 0 and split_col is None:
        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        enum = sss.split(X, y)
    elif split_col is not None and split_col in list(X_orig):
        sss = LeaveOneGroupOut()
        enum = sss.split(X, y, groups=X_orig[split_col])
        folds = len(X_orig[split_col].unique())
        if split_col in list(X): X.drop(labels=split_col, axis=1, inplace=True)
    else:
        raise ValueError(f"Improper number of folds: {folds}")
    features = list(X)
    hyperparams_list = get_params_list(params)
    hyperparams_perf = {}
    for hyper_idx, _ in enumerate(hyperparams_list):
        hyperparams_perf[hyper_idx] = []
    print("Starting Internal %d-fold validation for Set:%d" % (folds, set_id))
    # run all folds to find best hyperparameters

    import ipdb;
    ipdb.set_trace()
    print(X.shape)
    print(y.shape)
    for i, (train_index, test_index) in enumerate(enum, start=1):
        # t0 = time.time()
        print("Starting Split %d" % (i))
        X_train_val, y_train_val = X.iloc[train_index], y.iloc[train_index]
        # X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(), test_size=1.0/(folds-1), random_state=seed)
        X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(), test_size=0.1,
                                                        random_state=seed)
        X_train, X_val, y_train, y_val = X.loc[X_train_idx].to_numpy(), X.loc[X_val_idx].to_numpy(), y.loc[
            X_train_idx].to_numpy().flatten(), y.loc[X_val_idx].to_numpy().flatten()
        idxs_fold = idxs.loc[X_val_idx]
        # random under sampling if wanted
        if under_sample:
            # rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=seed)
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        # evaluate all hyperparameter combinations for this fold
        best_auc = -np.inf
        # best_y_preds, best_fi, best_params_idx, best_params = None, None, None, None
        for params_idx, next_params in enumerate(hyperparams_list):
            # create model, fit, predict, log predictions, log FIs
            model = XGBClassifier(**next_params, use_label_encoder=False)
            model.fit(X_train, y_train, verbose=0)  # warning: creating an ndarray from ragged nested sequences
            y_preds = model.predict_proba(X_val)[:, 1]
            fi = model.feature_importances_
            preds.append_predictions_DF(idxs_fold, y_val, y_preds, label=label, fold=i, hyperparams_idx=params_idx,
                                        hyperparams=next_params)
            fis.append_importances_DF(label, features, fi, fold=i, hyperparams_idx=params_idx, hyperparams=next_params)
            # keep track of this fold's best stats to log with params idx = -1 after all are evaluated
            try:
                this_auc = roc_auc_score(y_val, y_preds)
            except:
                this_auc = -np.inf
            hyperparams_perf[params_idx].append(this_auc)
            print(
                f"Fold: {i}, Hyperparams Index: {params_idx}, AUC Performance: {this_auc}, \n\t\tHyperparams: {next_params}")
            # if this_auc > best_auc:
            #    best_y_preds, best_fi, best_params_idx, best_params, best_model, best_auc = \
            #        y_preds, fi, params_idx, next_params, model, this_auc

    # identify best hyperparameters            
    best_hyperparams_avg_perf = -np.inf
    best_hyper_idx = None
    for check_hyper_idx in list(hyperparams_perf.keys()):
        next_avg = mean(hyperparams_perf[check_hyper_idx])
        if next_avg > best_hyperparams_avg_perf:
            best_hyperparams_avg_perf = next_avg
            best_hyper_idx = check_hyper_idx
    best_params = hyperparams_list[best_hyper_idx]
    print(f"\tbest params: index ({best_hyper_idx}) - {best_params}")

    # test best hyperparameter set on all folds
    if folds > 0 and split_col is None:
        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        enum = sss.split(X, y)
    elif split_col is not None and split_col in list(X_orig):
        sss = LeaveOneGroupOut()
        enum = sss.split(X, y, groups=X_orig[split_col])
        folds = len(X_orig[split_col].unique())
        if split_col in list(X): X.drop(labels=split_col, axis=1, inplace=True)
    else:
        raise ValueError(f"Improper number of folds: {folds}")

    best_auc, best_fold = -np.inf, None
    for i, (train_index, test_index) in enumerate(enum, start=1):
        X_train_val, y_train_val = X.iloc[train_index].to_numpy(), y.iloc[train_index].to_numpy()
        X_test, y_test = X.iloc[test_index].to_numpy(), y.iloc[test_index].to_numpy()
        idxs_fold = idxs.loc[test_index]
        if under_sample:
            # rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=seed)
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
            X_train_val, y_train_val = rus.fit_resample(X_train_val, y_train_val)
        model = XGBClassifier(**best_params, use_label_encoder=False)
        model.fit(X_train_val, y_train_val, verbose=0)
        y_preds = model.predict_proba(X_test)[:, 1]
        fi = model.feature_importances_
        try:
            this_auc = roc_auc_score(y_test, y_preds)
        except:
            this_auc = -np.inf
        preds.append_predictions_DF(idxs_fold, y_test, y_preds, label=label, fold=i, hyperparams_idx=-1,
                                    hyperparams=best_params)
        fis.append_importances_DF(label, features, fi, fold=i, hyperparams_idx=-1, hyperparams=best_params)
        if save_model:
            # pickle and save the automl object
            if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
            f_name = save_model_pth + '/xgboost_model_set_' + str(set_id) + '_fold_' + str(i) + '.pkl'
            with open(f_name, 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print("Fold %d Test AUROC: %f" % (i, this_auc))
        if this_auc > best_auc:
            best_auc = this_auc
            best_model = model
        print(f"Best fold: {best_fold}, Best AUC: {best_auc}")
        # print(f"\tbest params: index ({best_hyper_idx}) - {best_params}")
        # print(f"\tbest estimator: {model}")
        # print("Fold time: ", time.time()-t0)

    return preds, fis, hyperparams_list[best_hyper_idx], best_model


def eval_xgboost_kfold_grid(dataframes_list, preds, fis, params, set_id=1, label="", normalize=False, folds=10,
                            id_col="", save_model=True, save_model_pth='./', seed=0, under_sample=False):
    print("Starting internal xgboost k-fold validation...")
    if len(dataframes_list) != 2:
        raise ValueError(f"Length of dataframes_list must be 2, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y = dataframes_list[0], dataframes_list[1]
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index
    if normalize:
        X = normalize_df(X)  # [0,1]
    features = list(X)

    print("Feature Shape Internal: ", X.shape)
    print("Label Shape Internal", y.shape)

    # get the iterator
    if folds > 0:
        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)  # folds
        enum = sss.split(X, y)
    else:
        raise ValueError(f"Improper number of folds: {folds}")

    # generate a list of all paramter combinations from the params dict
    hyperparams_list = get_params_list(params)
    hyperparams_perf = {}  # create a hashmap to track the performance of each hyperparameter set in each fold
    for hyper_idx, _ in enumerate(hyperparams_list):
        hyperparams_perf[hyper_idx] = []

    i = 1
    print("Starting Internal %d-fold validation for Set:%d" % (folds, set_id))
    # run all folds
    for train_index, test_index in enum:  # iterate over the folds
        t0 = time.time()
        print("Starting Split %d" % (i))
        # parse out the data, labels, split them and convert to numpy arrays
        X_train_val, y_train_val = X.iloc[train_index], y.iloc[train_index]
        X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(),
                                                        test_size=1.0 / (folds - 1), random_state=seed)
        X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        idxs_fold = idxs.loc[test_index]
        X_train, y_train, X_val, y_val, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
        # random under sampling if wanted - not default
        if under_sample:
            # rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=seed)
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        # evaluate all hyperparameter combinations for this fold
        best_auc = -np.inf  # track the best auc for this fold
        best_y_preds, best_fi, best_params_idx, best_params = None, None, None, None  # some flags for tracking AUC performance for each fold
        for params_idx, next_params in enumerate(hyperparams_list):  # all parameter combinations
            # create model, fit, predict, log predictions, log FIs
            model = XGBClassifier(**next_params, use_label_encoder=False)  # define model
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      verbose=0)  # train model on train set and optimize on validation
            y_preds = model.predict_proba(X_test)[:, 1]  # make predictions on test set
            fi = model.feature_importances_  # track feature importances
            preds.append_predictions_DF(idxs_fold, y_test, y_preds, label=label, fold=i, hyperparams_idx=params_idx,
                                        hyperparams=next_params)
            fis.append_importances_DF(label, features, fi, fold=i, hyperparams_idx=params_idx, hyperparams=next_params)
            # keep track of this fold's best stats to log with params idx = -1 after all are evaluated
            this_auc = roc_auc_score(y_test, y_preds)
            hyperparams_perf[params_idx].append(this_auc)
            if this_auc > best_auc:
                best_y_preds, best_fi, best_params_idx, best_params, best_model, best_auc = \
                    y_preds, fi, params_idx, next_params, model, this_auc
        preds.append_predictions_DF(idxs_fold, y_test, best_y_preds, label=label, fold=i, hyperparams_idx=-1,
                                    hyperparams=best_params)
        fis.append_importances_DF(label, features, best_fi, fold=i, hyperparams_idx=-1, hyperparams=best_params)

        if save_model:
            # pickle and save the automl object
            if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
            f_name = save_model_pth + '/xgboost_model_set_' + str(set_id) + '_fold_' + str(i) + '.pkl'
            with open(f_name, 'wb') as f:
                pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
        print("Fold %d Test AUROC: %f" % (i, best_auc))
        print(f"\tbest params: index ({best_params_idx}) - {best_params}")
        print(f"\tbest estimator: {best_model}")
        print("Fold time: ", time.time() - t0)
        i += 1
    # identify best overall parameters
    best_hyperparams_avg_perf = -np.inf
    best_hyper_idx = None
    for check_hyper_idx in list(hyperparams_perf.keys()):
        next_avg = mean(hyperparams_perf[check_hyper_idx])
        if next_avg > best_hyperparams_avg_perf:
            best_hyperparams_avg_perf = next_avg
            best_hyper_idx = check_hyper_idx

    return preds, fis, hyperparams_list[best_hyper_idx]


def eval_xgboost_test_grid(dataframes_list, preds, fis, params, set_id=1, label="", normalize=True, id_col="",
                           save_model=False, save_model_pth='./', seed=0, under_sample=False, pretrained_model=None):
    print("Starting external xgboost validation...")
    if len(dataframes_list) != 4:
        raise ValueError(f"Length of dataframes_list must be 4, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y, X_ext, y_ext = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3]
    if len(list(X)) != len(list(X_ext)): raise ValueError("Internal and external sets have differing # of feats")
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index

    # drop ID Col if present: external
    if id_col in list(X_ext):
        idxs_ext = X_ext.loc[:, id_col]
        X_ext.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs_ext = X_ext.index

    # normalize internal and external data if wanted
    if normalize:
        X = normalize_df(X)
        X_ext = normalize_df(X_ext)

    print("Feature Count Internal: ", X.shape)
    print("Label Count Internal", y.shape)

    print("Feature Shape External: ", X_ext.shape)
    print("Label Shape Internal", y_ext.shape)

    features = list(X)

    t0 = time.time()
    # X_train_val, y_train_val = X.loc[train_index], y.loc[train_index]
    if pretrained_model is None:
        X_train_idx, X_val_idx, _, _ = train_test_split(X.index, y.to_numpy(), test_size=0.1, random_state=seed)
        X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]
        X_train, X_val, y_train, y_val = X_train.to_numpy(), X_val.to_numpy(), y_train.to_numpy(), y_val.to_numpy()
        X_ext, y_ext = X_ext.to_numpy(), y_ext.to_numpy()
    # params_fixed.update(best_params)
    if under_sample:
        # rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=seed)
        rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    if pretrained_model is None:
        model = XGBClassifier(**params, use_label_encoder=False)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    else:
        model = pretrained_model

    if save_model:
        # pickle and save the automl object
        if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
        f_name = save_model_pth + '/xgboost_model_set_' + str(set_id) + '_external' + '.pkl'
        with open(f_name, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    y_preds = model.predict_proba(X_ext)[:, 1]
    fi = model.feature_importances_
    preds.append_predictions_DF(idxs_ext, y_ext, y_preds, label=label, fold=-1, hyperparams_idx=-1, hyperparams=params)
    fis.append_importances_DF(label, features, fi, fold=i, hyperparams_idx=-1, hyperparams=params)
    print("Params set for external validation:", params)
    print(f"Best Estimator from external validation...: {model}")
    # def append_predictions_DF(self, indices, y_true, y_score, label, censoring=None, optional_dict=None,fold=0,hyperparams_idx=0,fus=None):
    roc_auc = roc_auc_score(y_ext, y_preds)
    print("External validation AUROC: %f" % (roc_auc))
    return preds, fis


def eval_xgboost_kfold(dataframes_list, preds, fis, params_fixed, params_grid, set_id=1, label="", normalize=True,
                       folds=10, id_col="", save_model=False, save_model_pth='./', seed=0, under_sample=False):
    print("Starting internal xgboost k-fold validation...")
    if len(dataframes_list) != 2:
        raise ValueError(f"Length of dataframes_list must be 2, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y = dataframes_list[0], dataframes_list[1]
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index
    if normalize:
        X = normalize_df(X)
    features = list(X)

    print("Feature Shape Internal: ", X.shape)
    print("Label Shape Internal", y.shape)

    if folds > 0:
        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        enum = sss.split(X, y)
    else:
        raise ValueError(f"Improper number of folds: {folds}")

    i = 1
    print("Starting Internal %d-fold validation for Set:%d" % (folds, set_id))
    for train_index, test_index in enum:
        t0 = time.time()
        print("Starting Split %d" % (i))
        X_train_val, y_train_val = X.iloc[train_index], y.iloc[train_index]
        X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(),
                                                        test_size=1.0 / (folds - 1), random_state=seed)
        X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        # X_train, X_test = X.loc[train_index],  X.loc[test_index]
        # y_train, y_test = y.loc[train_index], y.loc[test_index]
        idxs_fold = idxs.loc[test_index]

        estimator = XGBClassifier(**params_fixed, use_label_encoder=False)
        X_train_val, y_train_val = X_train_val.to_numpy(), y_train_val.to_numpy()
        mf = GridSearchCV(estimator=estimator,
                          param_grid=params_grid,
                          scoring='roc_auc',
                          cv=5,
                          n_jobs=4,
                          verbose=0)
        if under_sample:
            # rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=seed)
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
            X_train_val, y_train_val = rus.fit_resample(X_train_val, y_train_val)
        mf.fit(X_train_val, y_train_val, verbose=0)
        best_estimator = mf.best_estimator_
        fi = best_estimator.feature_importances_
        best_params = mf.best_params_
        # model.fit(X_train_val.to_numpy(),y_train_val.to_numpy(),eval_set=[(X_val.to_numpy(),y_val.to_numpy())],verbose=0)
        # fi = model.feature_importances_
        print(f"Best Params Set From K-Fold Grid Search: {best_params}")
        print(f"Best Estimator from fold...: {best_estimator}")
        # fold feature importances
        fis.append_importances_DF(label, features, fi, fold=i)

        if save_model:
            # pickle and save the automl object
            if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
            f_name = save_model_pth + '/xgboost_model_set_' + str(set_id) + '_fold_' + str(i) + '.pkl'
            with open(f_name, 'wb') as f:
                pickle.dump(best_estimator, f, pickle.HIGHEST_PROTOCOL)

        y_preds = best_estimator.predict_proba(X_test.to_numpy())[:, 1]
        # def append_predictions_DF(self, indices, y_true, y_score, label, censoring=None, optional_dict=None,fold=0,hyperparams_idx=0,fus=None):
        preds.append_predictions_DF(idxs_fold, y_test, y_preds, label=label, fold=i)
        roc_auc = roc_auc_score(y_test, y_preds)
        print("Fold %d Test AUROC: %f" % (i, roc_auc))
        print("Fold time: ", time.time() - t0)
        i += 1
    return preds, fis, params_fixed


def eval_xgboost_test(dataframes_list, preds, fis, params_fixed, params_grid, set_id=1, label="", normalize=True,
                      id_col="", save_model=False, save_model_pth='./', seed=0, under_sample=False):
    print("Starting external xgboost validation...")
    if len(dataframes_list) != 4:
        raise ValueError(f"Length of dataframes_list must be 4, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y, X_ext, y_ext = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3]
    if len(list(X)) != len(list(X_ext)): raise ValueError("Internal and external sets have differing # of feats")
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index

    # drop ID Col if present: external
    if id_col in list(X_ext):
        idxs_ext = X_ext.loc[:, id_col]
        X_ext.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs_ext = X_ext.index

    # normalize internal and external data if wanted
    if normalize:
        X = normalize_df(X)
        X_ext = normalize_df(X_ext)

    print("Feature Count Internal: ", X.shape)
    print("Label Count Internal", y.shape)

    print("Feature Shape External: ", X_ext.shape)
    print("Label Shape Internal", y_ext.shape)

    features = list(X)

    print("Starting External Validation %d" % (i))
    # X_train_val, y_train_val = X.loc[train_index], y.loc[train_index]
    # X_train_idx, X_val_idx, _, _ = train_test_split(X.index, y.to_numpy(), test_size=0.1, random_state=seed)
    # X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]

    print("Params fixed: ", params_fixed)
    print("Params grid: ", params_grid)
    # params_fixed.update(best_params)
    estimator = XGBClassifier(**params_fixed, use_label_encoder=False)
    X, y = X.to_numpy(), y.to_numpy()
    mf = GridSearchCV(estimator=estimator,
                      param_grid=params_grid,
                      scoring='roc_auc',
                      cv=5,
                      n_jobs=4,
                      verbose=0)
    if under_sample:
        # rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority',random_state=seed)
        rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
        X, y = rus.fit_resample(X, y)
    mf.fit(X, y, verbose=0)
    best_estimator = mf.best_estimator_
    fi = best_estimator.feature_importances_
    best_params = mf.best_params_
    params_fixed.update(best_params)
    print("Params set for external validation:", params_fixed)
    print(f"Best Estimator from external validation...: {best_estimator}")
    fis.append_importances_DF(label, features, fi, fold=-1)

    if save_model:
        # pickle and save the automl object
        if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
        f_name = save_model_pth + '/xgboost_model_set_' + str(set_id) + '_external' + '.pkl'
        with open(f_name, 'wb') as f:
            pickle.dump(best_estimator, f, pickle.HIGHEST_PROTOCOL)

    y_preds = best_estimator.predict_proba(X_ext.to_numpy())[:, 1]

    # def append_predictions_DF(self, indices, y_true, y_score, label, censoring=None, optional_dict=None,fold=0,hyperparams_idx=0,fus=None):
    preds.append_predictions_DF(idxs_ext, y_ext, y_preds, label=label, fold=-1)
    roc_auc = roc_auc_score(y_ext, y_preds)
    print("Test AUROC: %f" % (roc_auc))
    return preds, fis


def eval_flaml_kfold(dataframes_list, preds, fis, automl_settings, set_id=1, label="", normalize=True, folds=10,
                     id_col="", save_model=False, save_model_pth='./', seed=0):
    print("Starting internal k-fold validation...")
    if len(dataframes_list) != 2:
        raise ValueError(f"Length of dataframes_list must be 2, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y = dataframes_list[0], dataframes_list[1]
    features = list(X)
    if id_col in features:
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index
    if normalize:
        X = normalize_df(X)

    print("Feature Shape Internal: ", X.shape)
    print("Label Shape Internal", y.shape)

    if folds > 0:
        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        enum = sss.split(X, y)
    else:
        raise ValueError(f"Improper number of folds: {folds}")

    # best_params = mf.best_params_
    # Best Params Set From K-Fold Grid Search: {'colsample_bylevel': 0.3, 'colsample_bytree': 0.3, 'eta': 0.001, 'gamma': 0.25, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 250, 
    # params_fixed.update(best_params)

    i = 0
    print("Starting Internal %d-fold validation for Set:%d" % (folds, set_id))
    for train_index, test_index in enum:
        t0 = time.time()
        print("Starting Split %d" % (i))
        X_train_val, y_train_val = X.iloc[train_index], y.iloc[train_index]
        X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(),
                                                        test_size=1.0 / (folds - 1), random_state=seed)
        X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        # X_train, X_test = X.loc[train_index],  X.loc[test_index]
        # y_train, y_test = y.loc[train_index], y.loc[test_index]
        idxs_fold = idxs.loc[test_index]

        mf = AutoML()
        mf.fit(X_train_val.to_numpy(), y_train_val.squeeze().to_numpy(), **automl_settings)
        # print(mf.best_estimator)
        best_params = mf.best_config
        fi = mf.feature_importances_
        fis.append_importances_DF(label, features, fi, fold=i)
        best_estimator = mf.best_estimator
        print('Best ML learner:', best_estimator)
        print(f"Best Params Set From K-Fold Grid Search: {best_params}")
        print('Training duration of best run: {0:.4g}'.format(mf.best_config_train_time))

        y_preds = mf.predict_proba(X_test.to_numpy())[:, 1]
        # def append_predictions_DF(self, indices, y_true, y_score, label, censoring=None, optional_dict=None,fold=0,hyperparams_idx=0,fus=None):
        preds.append_predictions_DF(idxs_fold, y_test, y_preds, label=label, fold=i)
        roc_auc = roc_auc_score(y_test, y_preds)
        print("Fold %d Test AUROC: %f" % (i, roc_auc))
        t1 = time.time()
        print(f"Fold time: {t1 - t0}")
        i += 1
        if save_model:
            # pickle and save the automl object
            if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
            f_name = save_model_pth + '/automl_model_set_' + str(set_id) + '_fold_' + str(i) + '.pkl'
            with open(f_name, 'wb') as f:
                pickle.dump(mf, f, pickle.HIGHEST_PROTOCOL)
    return preds, fis


def eval_flaml_test(dataframes_list, preds, fis, automl_settings, set_id=1, label="", normalize=True, id_col="",
                    save_model=False, save_model_pth='./'):
    print("Starting external validation...")
    if len(dataframes_list) != 4:
        raise ValueError(f"Length of dataframes_list must be 4, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y, X_ext, y_ext = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3]

    # drop ID Col if present: internal
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index

    # drop ID Col if present: external
    if id_col in list(X_ext):
        idxs_ext = X_ext.loc[:, id_col]
        X_ext.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs_ext = X_ext.index

    # normalize internal and external data if wanted
    if normalize:
        X = normalize_df(X)
        X_ext = normalize_df(X_ext)

    print("Feature Count Internal: ", X.shape)
    print("Label Count Internal", y.shape)

    print("Feature Shape External: ", X_ext.shape)
    print("Label Shape Internal", y_ext.shape)
    print("Starting External Validation %d" % (i))
    features = list(X)

    # find optimal hyperparams with FLAML
    mf = AutoML()
    mf.fit(X.to_numpy(), y.squeeze().to_numpy(), **automl_settings)
    # print(mf.best_estimator)
    best_params = mf.best_config
    fi = mf.feature_importances_
    fis.append_importances_DF(label, features, fi, fold=-1)
    best_estimator = mf.best_estimator
    print('Best ML learner:', best_estimator)
    print(f"Best Params Set From K-Fold Grid Search: {best_params}")
    print('Training duration of best run: {0:.4g}'.format(mf.best_config_train_time))

    y_preds = mf.predict_proba(X_ext.to_numpy())[:, 1]
    # def append_predictions_DF(self, indices, y_true, y_score, label, censoring=None, optional_dict=None,fold=0,hyperparams_idx=0,fus=None):

    if save_model:
        # pickle and save the automl object
        if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
        f_name = save_model_pth + '/automl_model_set_' + str(set_id) + '_external' + '.pkl'
        with open(f_name, 'wb') as f:
            pickle.dump(mf, f, pickle.HIGHEST_PROTOCOL)

    preds.append_predictions_DF(idxs_ext, y_ext, y_preds, label=label, fold=-1)
    roc_auc = roc_auc_score(y_ext, y_preds)
    print("Test AUROC: %f" % (roc_auc))
    return preds, fis


def eval_optuna_kfold(dataframes_list, preds, fis, automl_settings, set_id=1, label="", normalize=True, folds=10,
                      id_col="", save_model=False, save_model_pth='./'):
    print("Starting internal k-fold validation...")
    if len(dataframes_list) != 2:
        raise ValueError(f"Length of dataframes_list must be 2, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y = dataframes_list[0], dataframes_list[1]
    features = list(X)
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index
    if normalize:
        X = normalize_df(X)

    print("Feature Shape Internal: ", X.shape)
    print("Label Shape Internal", y.shape)

    if folds > 0:
        sss = RepeatedStratifiedKFold(n_splits=folds, n_repeats=1, random_state=0)
        enum = sss.split(X, y)
    else:
        raise ValueError(f"Improper number of folds: {folds}")

    # best_params = mf.best_params_
    # Best Params Set From K-Fold Grid Search: {'colsample_bylevel': 0.3, 'colsample_bytree': 0.3, 'eta': 0.001, 'gamma': 0.25, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 250, 
    # params_fixed.update(best_params)

    i = 0
    features = list(X)
    print("Starting Internal %d-fold validation for Set:%d" % (folds, set_id))
    for train_index, test_index in enum:
        t0 = time.time()
        print("Starting Split %d" % (i))
        X_train_val, y_train_val = X.iloc[train_index], y.iloc[train_index]
        X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(),
                                                        test_size=1.0 / (folds - 1), random_state=42)
        X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        # X_train, X_test = X.loc[train_index],  X.loc[test_index]
        # y_train, y_test = y.loc[train_index], y.loc[test_index]
        idxs_fold = idxs.loc[test_index]
        # new optuna implementation
        if automl_settings['objective_func'] == 'log_loss':
            study = optuna.create_study(direction='minimize')
        elif automl_settings['objective_func'] == 'roc':
            study = optuna.create_study(direction='maximize')
        else:
            raise ValueError(f"unrecognized objective: {automl_settings['objective_func']}")
        ot = OptunaTest(X_train, X_val, y_train, y_val, features, automl_settings['objective_func'])
        study.optimize(ot.objective, n_trials=automl_settings['n_trials'], timeout=automl_settings['timeout'])
        # study.optimize(ot.objective, n_trials=automl_settings['n_trials'], timeout=automl_settings['timeout'], callbacks=[ot.callback])
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        best_params = trial.params
        print("  Value: {}".format(trial.value))
        print("  Params: {}".format(best_params))
        # mf = study.user_attrs["best_booster"] # booster() from trained xgboost model
        # dtrain=xgb.DMatrix(X_train, label=y_train, feature_names=features)
        # mf=xgb.train(best_params, dtrain)
        mf = XGBClassifier(**best_params, use_label_encoder=False)
        mf.fit(X_train_val, y_train_val)
        print(mf)
        fi_dict = mf.get_booster().get_score()
        print('FEATURE IMPORTANCES:')
        print(fi_dict)
        fi = [fi_dict[x] for x in features if x in list(fi_dict.keys())]
        fi_feats = [x for x in features if x in list(fi_dict.keys())]
        fis.append_importances_DF(label, fi_feats, fi, fold=i)
        # print('Best ML learner:',best_estimator)
        # print(f"Best Params Set From K-Fold Grid Search: {best_params}")
        # y_preds = mf.predict(X_test.to_numpy())[:,1]
        # dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
        # dtest =  xgb.DMatrix(X_test, feature_names=features)
        # y_preds = mf.predict(dtest)
        y_preds = mf.predict_proba(X_test)[:, 1]

        preds.append_predictions_DF(idxs_fold, y_test, y_preds, label=label, fold=i)
        roc_auc = roc_auc_score(y_test, y_preds)
        print("Fold %d Test AUROC: %f" % (i, roc_auc))
        t1 = time.time()
        print(f"Fold time: {t1 - t0}")
        i += 1
        if save_model:
            # pickle and save the automl object
            if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
            f_name = save_model_pth + '/automl_model_set_' + str(set_id) + '_fold_' + str(i) + '.pkl'
            with open(f_name, 'wb') as f:
                pickle.dump(mf, f, pickle.HIGHEST_PROTOCOL)
    return preds, fis


def eval_optuna_test(dataframes_list, preds, fis, automl_settings, set_id=1, label="", normalize=True, id_col="",
                     save_model=False, save_model_pth='./'):
    print("Starting external validation...")
    if len(dataframes_list) != 4:
        raise ValueError(f"Length of dataframes_list must be 4, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y, X_ext, y_ext = dataframes_list[0], dataframes_list[1], dataframes_list[2], dataframes_list[3]

    features = list(X)
    # drop ID Col if present: internal
    if id_col in list(X):
        idxs_train = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index

    # drop ID Col if present: external
    if id_col in list(X_ext):
        idxs_ext = X_ext.loc[:, id_col]
        X_ext.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs_ext = X_ext.index

    # normalize internal and external data if wanted
    if normalize:
        X = normalize_df(X)
        X_ext = normalize_df(X_ext)

    folds = 1
    if folds > 0:
        sss = RepeatedStratifiedKFold(n_splits=folds, n_repeats=1, random_state=0)
        enum = sss.split(X, y)
    else:
        raise ValueError(f"Improper number of folds: {folds}")

    print("Feature Count Internal: ", X.shape)
    print("Label Count Internal", y.shape)

    print("Feature Shape External: ", X_ext.shape)
    print("Label Shape Internal", y_ext.shape)
    print("Starting External Validation %d" % (i))

    i = -1
    for train_index, test_index in enum:  # is only going to pass through this one time
        t0 = time.time()
        print("Starting external validation %d" % (i))
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        # X_train_idx, X_val_idx, _, _ = train_test_split(X_train_val.index, y_train_val.to_numpy(), test_size=1.0/(folds-1), random_state=42)
        # X_train, X_val, y_train, y_val = X.loc[X_train_idx], X.loc[X_val_idx], y.loc[X_train_idx], y.loc[X_val_idx]
        X_val, y_val = X.iloc[test_index], y.iloc[test_index]
        idxs_fold = idxs.loc[test_index]

        # new optuna implementation
        study = optuna.create_study(direction='minimize')
        ot = OptunaTest(X_train, X_val, y_train, y_val, features)
        study.optimize(ot.objective, n_trials=automl_settings['n_trials'], timeout=automl_settings['timeout'])
        # study.optimize(ot.objective, n_trials=automl_settings['n_trials'], timeout=automl_settings['timeout'], callbacks=[ot.callback])
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        best_params = trial.params
        print("  Value: {}".format(trial.value))
        print("  Params: {}".format(best_params))
        # mf = study.user_attrs["best_booster"] # booster() from trained xgboost model
        # dtrain=xgb.DMatrix(X_train, label=y_train, feature_names=features)
        # mf=xgb.train(best_params, dtrain)
        mf = XGBClassifier(**best_params, use_label_encoder=False)
        mf.fit(X, y)
        print(mf)
        fi_dict = mf.get_booster().get_score()
        print('FEATURE IMPORTANCES:')
        print(fi_dict)
        fi = [fi_dict[x] for x in features if x in list(fi_dict.keys())]
        fi_feats = [x for x in features if x in list(fi_dict.keys())]
        fis.append_importances_DF(label, fi_feats, fi, fold=i)
        # print('Best ML learner:',best_estimator)
        # print(f"Best Params Set From K-Fold Grid Search: {best_params}")
        # y_preds = mf.predict(X_test.to_numpy())[:,1]
        # dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
        # dtest =  xgb.DMatrix(X_test, feature_names=features)
        # y_preds = mf.predict(dtest)
        y_preds = mf.predict_proba(X_ext)[:, 1]
        t1 = time.time()
        print(f"Fold time: {t1 - t0}")
        i += 1

    if save_model:
        # pickle and save the automl object
        if save_model_pth[-1] == '/': save_model_pth = save_model_pth[:-1]
        f_name = save_model_pth + '/automl_model_set_' + str(set_id) + '_fold_' + str(i) + '.pkl'
        with open(f_name, 'wb') as f:
            pickle.dump(mf, f, pickle.HIGHEST_PROTOCOL)

    preds.append_predictions_DF(idxs_ext, y_ext, y_preds, label=label, fold=-1)
    roc_auc = roc_auc_score(y_ext, y_preds)
    print("Test AUROC: %f" % (roc_auc))
    return preds, fis


class OptunaTest():
    def __init__(self, train_x, valid_x, train_y, valid_y, features, objective_func):
        # self.train_x = train_x.astype('float')
        # self.valid_x = valid_x.astype('float')
        # self.train_y = train_y.astype('float')
        # self.valid_y = valid_y.astype('float')
        self.train_x = train_x
        self.valid_x = valid_x
        self.train_y = train_y
        self.valid_y = valid_y
        self.features = features
        if objective_func in ['log_loss', 'roc']:
            self.objective_func = objective_func
        else:
            raise ValueError(f"Unrecognized objective: {objective_func}")

    def objective(self, trial):
        logger = optuna.logging.get_logger("optuna")
        logger.setLevel(optuna.logging.ERROR)
        # (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
        # train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
        dtrain = xgb.DMatrix(self.train_x, label=self.train_y, feature_names=self.features)
        dvalid = xgb.DMatrix(self.valid_x, label=self.valid_y, feature_names=self.features)

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            # "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 1e-1),
            'scale_pos_weight': trial.suggest_float("scale_pos_weight", 1e-1, 10),

        }
        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 2, 10, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        if self.objective_func == 'roc':
            num_boost_round = 500
            evallist = [(dvalid, 'validation')]
            model = xgb.train(params=param, dtrain=dtrain, evals=evallist,
                              num_boost_round=num_boost_round, early_stopping_rounds=10)
            preds = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
            pred_labels_val = np.rint(preds)
            roc_val = sklearn.metrics.roc_auc_score(self.valid_y, pred_labels_val)
            return roc_val
        elif self.objective_func == 'log_loss':
            model = XGBClassifier(**param, use_label_encoder=False)
            model.fit(self.train_x, self.train_y)
            preds = model.predict_proba(self.valid_x)
            score = sklearn.metrics.log_loss(self.valid_y, preds)
            return score
        else:
            raise ValueError(f"Unrecognized objective: {self.objective_func}")

    # def callback(self, study, trial):
    #     if study.best_trial.number == trial.number:
    #         study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def objective(trial, dtrain, dval, y_val):
    param = {'objective': 'binary:logistic',
             'eval_metric': 'logloss',
             'booster': 'gbtree',
             'max_depth': trial.suggest_int("max_depth", 2, 8),
             'eta': trial.suggest_loguniform("eta", 1e-8, 0.5),
             'min_child_weight': trial.suggest_float("min_child_weight", 0.5, 3),
             'colsample_bytree': trial.suggest_float("colsample_bytree", 0.4, 0.8),
             'subsample': trial.suggest_float("subsample", 0.4, 0.8),
             'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
             'scale_pos_weight': trial.suggest_loguniform("scale_pos_weight", 1e-8, 10),
             'lambda': trial.suggest_float("lambda", 0, 10),
             'alpha': trial.suggest_float("alpha", 0, 10),
             'num_parallel_tree': 1,
             'gamma': trial.suggest_float("gamma", 0, 1.0)
             }
    num_boost_round = 500
    evallist = [(dval, 'validation')]

    #     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    #     model = xgb.train(params = param, dtrain = dtrain, evals=evallist, num_boost_round = 50, 
    #                       early_stopping_rounds=10, callbacks=[pruning_callback])
    #     preds = model.predict(dval, ntree_limit=model.best_ntree_limit)
    model = xgb.train(params=param, dtrain=dtrain, evals=evallist,
                      num_boost_round=num_boost_round, early_stopping_rounds=10)
    preds = model.predict(dval, ntree_limit=model.best_ntree_limit)
    pred_labels_val = np.rint(preds)
    roc_val = roc_auc_score(y_val, pred_labels_val)
    return roc_val


def eval_static(dataframes_list, preds, fis, set_id=1, label="", id_col=""):
    print("Starting internal k-fold validation...")
    if len(dataframes_list) != 2:
        raise ValueError(f"Length of dataframes_list must be 2, got {len(dataframes_list)}")
    for i, df in enumerate(dataframes_list):
        dataframes_list[i] = df.reset_index(drop=True)
    X, y = dataframes_list[0], dataframes_list[1]
    features = list(X)
    if id_col in list(X):
        idxs = X.loc[:, id_col]
        X.drop(labels=id_col, axis=1, inplace=True)
    else:
        idxs = X.index

    print("Feature Shape Internal: ", X.shape)
    print("Label Shape Internal", y.shape)

    assert (X.shape[1] == 1)
    tpr, fpr, thresholds = roc_curve(y, X, pos_label=0)
    roc_auc = auc(fpr, tpr)
    # test_idx = range(0,len(y),1)
    preds.append_predictions_DF(idxs, y, X.iloc[:, 0], label=label, fold=-1, hyperparams_idx=-1,
                                hyperparams='univariable model')
    print("Internal Test AUROC for %s: %f" % (label, roc_auc))
    return preds, fis


def reset_params():
    """
    Returns reset fixed and grid-search parameters for XGBoost

    Parameters:
    ---------------
    None.

    Returns:
    test_idx (pandas Series) - Series of indicies from test data points
    y_test (pandas Series) - Series of test true values
    y_preds (numpy array) - a numpy array of shape array-like of shape (n_samples, 1)
        with the probability of each data example being of a given class.
    gr (object) - grid search CB object
    ---------------
    """
    params_fixed = {
        'objective': 'binary:logistic',
        'nthread': 28,
        'eval_metric': 'auc',
        'subsample': 0.8,
        'tree_method': 'hist',  # gpu_hist for GPU accel hist for CPU
        # 'scale_pos_weight': 5,
        'verbosity': 0,
        'booster': 'gbtree',
    }
    # params_grid = {
    #         'eta': [0.01],
    #         'gamma': [10],
    #         'min_child_weight': [6],
    #         'max_depth': [6],
    #         'subsample': [0.8],
    #         'colsample_bytree': [0.6],
    #         # 'colsample_bylevel': [0.3],
    #         'n_estimators':[500,600],
    #         }
    params_grid = {
        'eta': [0.01, 0.1, 0.3],
        'gamma': [0.5, 0],
        'min_child_weight': [4],
        'max_depth': [4, 6],
        'subsample': [0.3],
        'colsample_bytree': [0.3],
        'colsample_bylevel': [0.3],
        'n_estimators': [1500, 2000, 2500],
    }

    return params_fixed, params_grid
