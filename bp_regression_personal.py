import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from time import time
import logging
from tqdm import tqdm

# scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

# xgboost
import xgboost as xgb

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


########################################################################
# 載入單一 .h5 檔案，將資料轉成 pandas DataFrame
########################################################################
def load_data_to_df(h5_files, subject_id_from='filename'):
    """
    讀取多個 .h5 檔案，對每個檔案中的各個片段擷取以下資訊：
      - 個人資訊: (N, 4) => [age, gender, weight, height]
      - vascular_properties: (N, 3) => [ptt, pat, rr_interval]  
        （這裡 rr_interval 已經在檔案中儲存）
      - 血壓標籤: segsbp (sbp) 與 segdbp (dbp)
      - subjectID (以檔案名稱作為識別)
      
    回傳一個 pandas DataFrame。
    """
    logger.info(f"開始載入 {len(h5_files)} 個檔案...")
    start_time = time()
    rows = []
    
    for hf in h5_files:
        hfpath = Path(hf)
        if not hfpath.exists():
            logger.warning(f"檔案不存在: {hf}")
            continue

        try:
            with h5py.File(hfpath, 'r') as f:
                if 'personal_info' not in f:
                    logger.warning(f"跳過 {hf}, 無個人資訊")
                    continue

                # 讀取各項資料
                personal = f['personal_info'][:]         # shape (N, 4)
                vascular = f['vascular_properties'][:]     # shape (N, 3) ，依序為 [ptt, pat, rr_interval]
                sbp = f['segsbp'][:]                       # shape (N,)
                dbp = f['segdbp'][:]                       # shape (N,)
                
                N = len(sbp)
                subject_id = hfpath.stem if subject_id_from == 'filename' else "unknown"
                
                for i in range(N):
                    # 個人資訊（在單一受試者中應該皆相同）
                    age, gender, w, h = personal[i]
                    # vascular properties（若長度不足 3，補 0）
                    if len(vascular[i]) >= 3:
                        ptt, pat, rr = vascular[i][:3]
                    else:
                        ptt, pat = vascular[i][:2]
                        rr = 0.0
                    row = {
                        'age': float(age),
                        'gender': float(gender),
                        'weight': float(w),
                        'height': float(h),
                        'ptt': float(ptt),
                        'pat': float(pat),
                        'rr_interval': float(rr),
                        'sbp': float(sbp[i]),
                        'dbp': float(dbp[i]),
                        'subjectID': subject_id
                    }
                    rows.append(row)
        except Exception as e:
            logger.error(f"處理檔案 {hf} 時發生錯誤: {str(e)}")
            continue

    df = pd.DataFrame(rows)
    end_time = time()
    logger.info(f"載入完成! 總計 {len(df)} 筆片段, 耗時 {end_time - start_time:.2f} 秒")
    return df


########################################################################
# 分割資料：使用 GroupShuffleSplit 保證同一 subjectID 不會分散到 train/test
########################################################################
def split_train_test_by_subject(df, train_ratio=0.8):
    """
    使用 GroupShuffleSplit，以 subjectID 為單位做 train/test split。
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=42)
    groups = df['subjectID']
    for train_idx, test_idx in splitter.split(df, groups=groups):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


########################################################################
# 使用多種回歸模型進行訓練與評估，回傳每種模型的 MAE 與 R²
# 這裡我們採用以下模型：
#   - Linear Regression
#   - Random Forest
#   - XGBoost
#   - SVR
#   - Kernel Ridge Regression
########################################################################
def train_and_eval_regressors(X_train, y_train, X_test, y_test, label_name="SBP"):
    """
    給定四種回歸器(Linear, RF, XGBoost, SVR) + DummyMean，
    分別計算 MAE, R^2, 以及 (y_test - pred) 的絕對誤差之標準差 STD。
    回傳 results[model_name] = (mae, r2, std)
    """
    logger.info(f"開始訓練 {label_name} 模型，共 {len(X_train)} 筆訓練資料")
    results = {}
    
    # 1) 各種回歸器
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42,
                                     use_label_encoder=False, eval_metric='rmse'),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }
    for name, model in models.items():
        logger.info(f"  訓練 {name} ...")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            abs_err = np.abs(y_test - pred)
            mae = np.mean(abs_err)
            std = np.std(abs_err)
            r2 = r2_score(y_test, pred)
        except Exception as e:
            logger.error(f"{name} 訓練/預測錯誤: {e}")
            mae, r2, std = np.nan, np.nan, np.nan
        
        results[name] = (mae, r2, std)

    # 2) DummyMean => 用該人的訓練集平均值當輸出
    dummy_val = np.mean(y_train)
    pred_dummy = np.full_like(y_test, dummy_val, dtype=float)
    abs_err_dummy = np.abs(y_test - pred_dummy)
    mae_dummy = np.mean(abs_err_dummy)
    std_dummy = np.std(abs_err_dummy)
    r2_dummy = r2_score(y_test, pred_dummy)
    results["DummyMean"] = (mae_dummy, r2_dummy, std_dummy)

    # 3) 顯示結果
    logger.info(f"--- {label_name} 模型比較 ---")
    for key, (mae, r2, std) in results.items():
        logger.info(f"{key}: MAE = {mae:.3f}, STD = {std:.3f}, R² = {r2:.3f}")
    return results


########################################################################
# 主流程：針對每個人進行個人化回歸（僅使用 vascular_properties: ptt, pat, rr_interval）
# 並將每個受試者結果彙整成單一列：
#   欄位包含：subjectID, sample_count, age, gender, weight, height，
#   以及每種回歸方法的 SBP_MAE 與 DBP_MAE（例如：LinearRegression_SBP_MAE, LinearRegression_DBP_MAE, ...）
########################################################################
def main():
    # 個人化資料存放目錄（請確保此資料夾中每個 .h5 檔代表同一個 subject 的資料）
    data_dir = Path('personalized_training_data_VitalDB')
    if not data_dir.exists():
        logger.error(f"資料夾 {data_dir} 不存在！")
        return

    # 找出該資料夾中所有 .h5 檔案
    h5_files = sorted(data_dir.glob("*.h5"))
    if len(h5_files) == 0:
        logger.error("找不到任何個人化的 .h5 檔案")
        return
    logger.info(f"找到 {len(h5_files)} 個個人化資料檔案")

    results_list = []

    for hf in tqdm(h5_files, desc="處理個人化資料"):
        # 載入單一 subject 資料
        df = load_data_to_df([hf])
        if df.empty:
            logger.warning(f"{hf.name} 資料為空，跳過")
            continue

        subject_id = df['subjectID'].iloc[0]
        sample_count = len(df)
        logger.info(f"處理 subjectID: {subject_id}，片段數: {sample_count}")

        # 若資料筆數不足（例如少於 10 筆），則跳過
        if sample_count < 10:
            logger.warning(f"{subject_id} 片段數 {sample_count} 太少，跳過個人化回歸")
            continue

        # 個人化回歸僅用 vascular_properties：['ptt', 'pat', 'rr_interval']
        features = ['ptt', 'pat', 'rr_interval']
        X = df[features].values
        y_sbp = df['sbp'].values
        y_dbp = df['dbp'].values

        # 為了讓 SBP 與 DBP 使用相同的分割，先隨機產生一組 indices
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * 0.7)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_sbp_train = y_sbp[train_idx]
        y_sbp_test = y_sbp[test_idx]
        y_dbp_train = y_dbp[train_idx]
        y_dbp_test = y_dbp[test_idx]

        # 分別訓練 SBP 與 DBP 模型
        logger.info(f"Subject {subject_id} - SBP 回歸:")
        sbp_results = train_and_eval_regressors(X_train, y_sbp_train, X_test, y_sbp_test, label_name="SBP")
        logger.info(f"Subject {subject_id} - DBP 回歸:")
        dbp_results = train_and_eval_regressors(X_train, y_dbp_train, X_test, y_dbp_test, label_name="DBP")

        # 由於個人資訊對同一受試者而言固定，取第一筆作為代表
        personal_info = df.iloc[0][['age', 'gender', 'weight', 'height']].to_dict()

        # 建立該受試者結果列，包含 subjectID、sample_count、及個人資訊
        subject_row = {
            'subjectID': subject_id,
            'sample_count': sample_count,
            'age': personal_info['age'],
            'gender': personal_info['gender'],
            'weight': personal_info['weight'],
            'height': personal_info['height']
        }
        # 將每種模型的 SBP 與 DBP MAE 存入同一列（欄位名稱格式：Model_SBP_MAE、Model_DBP_MAE）
        for model_name in sbp_results.keys():
            mae_sbp, r2_sbp, std_sbp = sbp_results[model_name]
            mae_dbp, r2_dbp, std_dbp = dbp_results[model_name]
            subject_row[f"{model_name}_SBP_MAE"] = mae_sbp
            subject_row[f"{model_name}_SBP_STD"] = std_sbp
            subject_row[f"{model_name}_SBP_R2"] = r2_sbp
            subject_row[f"{model_name}_DBP_MAE"] = mae_dbp
            subject_row[f"{model_name}_DBP_STD"] = std_dbp
            subject_row[f"{model_name}_DBP_R2"] = r2_dbp

        results_list.append(subject_row)

    # 組成結果 DataFrame，每一列為一位受試者
    results_df = pd.DataFrame(results_list)
    logger.info("=== 個人化回歸結果 (含MAE, STD, R²) ===")
    logger.info(results_df)

    # 將結果存成 CSV 檔
    results_df.to_csv("personalized_regression_results.csv", index=False, float_format="%.3f")
    logger.info("結果已存成 personalized_regression_results.csv")

if __name__ == "__main__":
    main()
