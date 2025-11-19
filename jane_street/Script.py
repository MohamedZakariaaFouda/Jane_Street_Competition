# =================
# Import libraries
# =================
import os
import joblib 
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
import numpy as np 
import copy
from joblib import Parallel, delayed
import kaggle_evaluation.jane_street_inference_server

# =================
# Configuration
# ==================

# use the Kaggle input directory
input_path = '/kaggle/input/jane-street-realtime-marketdata-forecasting'

# Flag to determine if the script is in training mode or not
TRAINING = True

# Define the feature names based on the number of features (79 in this case)
feature_names = [f"feature_{i:02d}" for i in range(79)]

# Number of validation dates to use
num_valid_dates = 100

# Number of dates to skip from the beginning of the dataset
skip_dates = 1473

# Number of folds for cross-validation
N_fold = 5

# ============================
# Reduce Memory Usage Function
# ============================

def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        
        # Skip non-numeric columns
        if col_type.kind not in ['i', 'u', 'f']:
            continue
        
        c_min = df[col].min()
        c_max = df[col].max()

        # Integer types
        if col_type.kind in ['i', 'u']:
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # Float types
        else:
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float32 if float16_as32 else np.float16)
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%")

    return df


# ================================
#  Custom R2 Metrics for Models
# ===============================

def r2_xgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / \
              (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return -r2  # XGB wants a loss → lower is better


def r2_lgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / \
              (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return 'r2', r2, True  # higher = better


class r2_cbt(object):
    def get_final_error(self, error, weight):
        return 1 - error / (weight + 1e-38)

    def is_max_optimal(self):
        return True  # maximize

    def evaluate(self, approxes, target, weight):
        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w * (target[i] ** 2)
            error_sum += w * ((approx[i] - target[i]) ** 2)

        return error_sum, weight_sum


# ============================
#  Model Dictionary
# ============================

model_dict = {
    'lgb': lgb.LGBMRegressor(
        n_estimators=500,
        device='gpu',
        gpu_use_dp=True,
        objective='l2'
    ),

    'xgb': xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        device="cuda",
        objective='reg:squarederror',
        eval_metric=r2_xgb,
        disable_default_eval_metric=True
    ),

    'cbt': cbt.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        task_type='GPU',
        loss_function='RMSE',
        eval_metric=r2_cbt()
    ),
}

# ============================
#  Create model directory
# ============================
os.makedirs("models", exist_ok=True)

# =====================================
# Train and Evulate odels on N_Folds
# =====================================
def train_one_fold(model_dict, model_name, fold_id):
    print(f"\n==== Training {model_name} — Fold {fold_id} ====\n")

    # Select dates for training (K-fold by date index)
    selected_dates = [date for ii, date in enumerate(train_dates)if ii % N_fold != fold_id]

    # Training data
    X_train = df[feature_names].loc[df['date_id'].isin(selected_dates)]
    y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)]
    w_train = df['weight'].loc[df['date_id'].isin(selected_dates)]

    # Create a fresh copy of the model
    model = copy.deepcopy(model_dict[model_name])

    # LightGBM
    if model_name == "lgb":
        model.fit(
            X_train, y_train, w_train,
            eval_set=[(X_valid, y_valid, w_valid)],
            eval_metric=[r2_lgb],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(10)
            ]
        )

    # CatBoost
    elif model_name == "cbt":
        evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[evalset],
            verbose=10,
            early_stopping_rounds=100
        )

    # XGBoost
    else:
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            sample_weight_eval_set=[w_valid],
            verbose=10,
            early_stopping_rounds=100
        )

    # Save model
    joblib.dump(model, f"models/{model_name}_{fold_id}.model")

    del X_train, y_train, w_train
    gc.collect()

    return model


# ===================================
# Load and Perpare Data for Training 
# ===================================

# If in training mode, load the training data
if TRAINING:
    # Load the training data and Filter the DataFrame to include only dates >= skip_dates
    df = pd.read_parquet(f"{input_path}/train.parquet",filters=[("date_id", ">=", skip_dates)]).reset_index(drop=True)
    
    # Reduce memory usage of the DataFrame (function not provided here)
    df = reduce_mem_usage(df, False)
    
    # Get unique dates from the DataFrame
    dates = df['date_id'].unique()
    
    # Define validation dates as the last `num_valid_dates` dates
    valid_dates = dates[-num_valid_dates:]
    
    # Define training dates as all dates except the last `num_valid_dates` dates
    train_dates = dates[:-num_valid_dates]
    
    # Display the last few rows of the DataFrame (for debugging purposes)
    print(df.tail())
    print(f'shape of data{df.shape}')

# =========================================
#  Prepare Validation Set for Training Mode
# =========================================
if TRAINING:
    X_valid = df[feature_names].loc[df['date_id'].isin(valid_dates)]
    y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)]
    w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)]

# list to add our models
models = []
# =============================
#  Actiave Training Loop 
# =============================
if TRAINING:
    for fold in range(N_fold): 
        for model_name in model_dict.keys():  
            model = train_one_fold(model_dict, model_name, fold) 
            models.append(model)

# =============================
# Load Pretrained Models
# =============================
else:
    for fold in range(N_fold):
        for model_name in model_dict.keys():
            models.append(
                joblib.load(f"models/{model_name}_{fold}.model")
            )

# ========================================
# Prediction Using the Ensemble of Models
# ========================================

lags_: pl.DataFrame | None = None

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction using the ensemble of models."""
    global lags_
    if lags is not None:
        lags_ = lags

    # Convert features to NumPy for model prediction
    feat = test.select(feature_names).to_numpy()
    
    # Ensemble prediction (average over all models)
    pred = np.mean([model.predict(feat) for model in models], axis=0)
    
    # Create Polars DataFrame for submission
    predictions = pl.DataFrame({
        'row_id': test['row_id'],
        'responder_6': pred.astype(np.float32)
    })
    
    # Assertions for safety
    assert isinstance(predictions, pl.DataFrame)
    assert list(predictions.columns) == ['row_id', 'responder_6']
    assert len(predictions) == len(test)

    return predictions


#=================
# Submission
#=================

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/test.parquet',
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/lags.parquet',
        )
    )
