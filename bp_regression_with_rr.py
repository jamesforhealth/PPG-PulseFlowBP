import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import logging
from tqdm import tqdm

# scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
#from sklearn.model_selection import train_test_split  # 不再用此方法

# xgboost
import xgboost as xgb

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_to_df(h5_files, subject_id_from='filename'):
    """
    讀取多個 .h5 檔案，對每個檔案中的各個片段擷取以下資訊：
      - 個人資訊: (N, 4) => [age, gender, weight, height]
      - vascular_properties: (N, 3) => [ptt, pat, rr_interval]  
        （這裡 rr_interval 已經在檔案中儲存，代表三個與血壓高度相關的時間特徵）
      - 血壓標籤: segsbp (sbp) 與 segdbp (dbp)
      - subjectID (以檔案名稱作為識別)
      
    回傳一個 pandas DataFrame。
    """
    logger.info(f"開始載入 {len(h5_files)} 個檔案...")
    start_time = time()
    rows = []
    
    for hf in tqdm(h5_files, desc="載入檔案"):
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
                    # 個人資訊
                    age, gender, w, h = personal[i]
                    # vascular properties
                    ptt, pat, rr = vascular[i]
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


def train_and_eval_regressors(X_train, y_train, X_test, y_test, model_name="SBP"):
    """
    使用四種回歸模型（Linear Regression、RandomForest、XGBoost、MLP）
    分別訓練並評估效能，並印出 MAE 與 R²。
    """
    logger.info(f"\n=== 開始訓練 {model_name} 模型 ===")
    results = {}

    # 1) Linear Regression
    logger.info("1/4 訓練 Linear Regression...")
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)
    mae_lin = mean_absolute_error(y_test, pred_lin)
    r2_lin = r2_score(y_test, pred_lin)
    results["Linear"] = (mae_lin, r2_lin, lin)

    # 2) Random Forest
    logger.info("2/4 訓練 Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)
    results["RandomForest"] = (mae_rf, r2_rf, rf)

    # 3) XGBoost
    logger.info("3/4 訓練 XGBoost...")
    xgbr = xgb.XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
    xgbr.fit(X_train, y_train)
    pred_xgb = xgbr.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, pred_xgb)
    r2_xgb = r2_score(y_test, pred_xgb)
    results["XGBoost"] = (mae_xgb, r2_xgb, xgbr)

    # 4) MLP
    logger.info("4/4 訓練 MLP...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    r2_mlp = r2_score(y_test, pred_mlp)
    results["MLP"] = (mae_mlp, r2_mlp, mlp)

    logger.info(f"\n=== {model_name} 模型比較 ===")
    for key, (mae, r2, _) in results.items():
        logger.info(f"{key}: MAE = {mae:.3f}, R² = {r2:.3f}")

    return results


def main():
    """
    主流程：
      1. 從指定資料夾中讀取訓練資料（training_*.h5）用於模型訓練
      2. 從 validation.h5 與 test.h5 讀取測試資料（來自不同受試者）
      3. 選擇特徵：個人資訊 (4 維) 與 vascular_properties (3 維)，總共 7 維特徵
      4. 分別針對 SBP 與 DBP 執行回歸模型訓練與評估
    """
    data_dir = Path('training_data_VitalDB_quality')
    
    # 1. 載入訓練資料：training_*.h5
    training_files = sorted(data_dir.glob("training_*.h5"))
    if len(training_files) == 0:
        logger.error("找不到任何 training_*.h5 檔案")
        return
    logger.info(f"找到 {len(training_files)} 個 training 檔案")
    df_train = load_data_to_df(training_files)
    if df_train is None or df_train.empty:
        logger.error("訓練資料讀取失敗或為空！")
        return
    logger.info(f"訓練資料集共 {len(df_train)} 筆, 欄位: {df_train.columns.tolist()}")

    # 2. 載入測試資料：validation.h5 與 test.h5（來自不同受試者）
    test_files = []
    val_file = data_dir / 'validation.h5'
    test_file = data_dir / 'test.h5'
    if val_file.exists():
        test_files.append(val_file)
    if test_file.exists():
        test_files.append(test_file)
    if len(test_files) == 0:
        logger.error("找不到 validation.h5 或 test.h5 檔案作為測試資料")
        return
    logger.info(f"找到 {len(test_files)} 個測試檔案")
    df_test = load_data_to_df(test_files)
    if df_test is None or df_test.empty:
        logger.error("測試資料讀取失敗或為空！")
        return
    logger.info(f"測試資料集共 {len(df_test)} 筆, 欄位: {df_test.columns.tolist()}")

    # 3. 定義特徵與標籤：  
    # 特徵包含：個人資訊 (age, gender, weight, height) 與 vascular_properties (ptt, pat, rr_interval)
    feature_cols = ['age', 'gender', 'weight', 'height', 'ptt', 'pat', 'rr_interval']
    X_train = df_train[feature_cols].values
    y_sbp_train = df_train['sbp'].values
    y_dbp_train = df_train['dbp'].values

    X_test = df_test[feature_cols].values
    y_sbp_test = df_test['sbp'].values
    y_dbp_test = df_test['dbp'].values

    logger.info(f"訓練集大小: {len(X_train)}，測試集大小: {len(X_test)}")

    # 4. 分別訓練並評估 SBP 與 DBP 模型
    logger.info("\n===== SBP 回歸 =====")
    _ = train_and_eval_regressors(X_train, y_sbp_train, X_test, y_sbp_test, model_name="SBP")

    logger.info("\n===== DBP 回歸 =====")
    _ = train_and_eval_regressors(X_train, y_dbp_train, X_test, y_dbp_test, model_name="DBP")


if __name__ == "__main__":
    main()
