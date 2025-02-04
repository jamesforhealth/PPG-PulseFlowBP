##################################################
# bp_regression_all_methods.py
# 血壓回歸方法實作 (含進度提示和時間估計)
##################################################
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import os
from time import time
from tqdm import tqdm
import logging

# scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# xgboost
import xgboost as xgb

# 設定logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

##################################################
# (A) 載入資料: personal_info(4) + vascular(2) + SBP/DBP
##################################################
def load_data_to_df(h5_files, subject_id_from='filename'):
    """
    讀取多個 .h5 檔案，對每個檔案的 N segments，擷取:
      - age, gender, weight, height
      - ptt, pat
      - sbp, dbp
    回傳 pandas.DataFrame
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
                    
                personal = f['personal_info'][:]
                vascular = f['vascular_properties'][:]
                sbp = f['segsbp'][:]
                dbp = f['segdbp'][:]
                
                N = len(sbp)
                subject_id = hfpath.stem if subject_id_from=='filename' else "unknown"
                
                for i in range(N):
                    age, gender, w, h = personal[i]
                    ptt, pat = vascular[i]
                    row = {
                        'age': float(age),
                        'gender': float(gender),
                        'weight': float(w),
                        'height': float(h),
                        'ptt': float(ptt),
                        'pat': float(pat),
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
    logger.info(f"載入完成! 總計 {len(df)} 個片段, 耗時 {end_time - start_time:.2f} 秒")
    return df

##################################################
# (B) 全域回歸: 多種模型 (Linear / RF / XGB / MLP)
##################################################
def train_and_eval_regressors(X_train, y_train, X_test, y_test, model_name="SBP"):
    """
    嘗試多種回歸器並評估效能
    """
    logger.info(f"\n開始訓練{model_name}模型...")
    results = {}
    
    # 1) Linear Regression
    logger.info("1/4 訓練 Linear Regression...")
    start_time = time()
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)
    mae_lin = mean_absolute_error(y_test, pred_lin)
    r2_lin = r2_score(y_test, pred_lin)
    results["Linear"] = (mae_lin, r2_lin, lin)
    logger.info(f"Linear完成，耗時: {time() - start_time:.2f}秒")
    
    # 2) Random Forest
    logger.info("2/4 訓練 Random Forest...")
    start_time = time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)
    results["RandomForest"] = (mae_rf, r2_rf, rf)
    logger.info(f"RandomForest完成，耗時: {time() - start_time:.2f}秒")
    
    # 3) XGBoost
    logger.info("3/4 訓練 XGBoost...")
    start_time = time()
    xgbr = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgbr.fit(X_train, y_train)
    pred_xgb = xgbr.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, pred_xgb)
    r2_xgb = r2_score(y_test, pred_xgb)
    results["XGBoost"] = (mae_xgb, r2_xgb, xgbr)
    logger.info(f"XGBoost完成，耗時: {time() - start_time:.2f}秒")
    
    # 4) MLP
    logger.info("4/4 訓練 MLP...")
    start_time = time()
    mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    r2_mlp = r2_score(y_test, pred_mlp)
    results["MLP"] = (mae_mlp, r2_mlp, mlp)
    logger.info(f"MLP完成，耗時: {time() - start_time:.2f}秒")
    
    # 輸出結果比較
    logger.info(f"\n=== {model_name} 模型比較 ===")
    for k,v in results.items():
        mae, r2, _ = v
        logger.info(f"{k}: MAE={mae:.3f}, R^2={r2:.3f}")
    
    return results

##################################################
# (C) 個人化校正
##################################################
def compute_personal_offset(model, X_personal, y_personal):
    """計算個人化偏移量"""
    preds = model.predict(X_personal)
    offset = np.mean(y_personal - preds)
    return offset

def apply_offset(model, offset, X):
    """應用個人化偏移量"""
    preds = model.predict(X)
    return preds + offset

##################################################
# (D) main
##################################################
def main():
    logger.info("=== 血壓回歸模型訓練開始 ===")
    
    # 1) 準備檔案路徑
    logger.info("\n1. 準備檔案路徑...")
    data_dir = Path('training_data_VitalDB_quality')
    h5_files = []
    for i in range(1,10):
        f = data_dir/f"training_{i}.h5"
        if f.exists():
            h5_files.append(f)
    val_file = data_dir/'validation.h5'
    test_file = data_dir/'test.h5'
    if val_file.exists():
        h5_files.append(val_file)
    if test_file.exists():
        h5_files.append(test_file)
    
    # 2) 載入資料
    logger.info("\n2. 載入資料...")
    df = load_data_to_df(h5_files)
    
    # 3) 準備特徵和標籤
    logger.info("\n3. 準備特徵和標籤...")
    feature_cols = ['age','gender','weight','height','ptt','pat']
    X = df[feature_cols].values
    y_sbp = df['sbp'].values
    y_dbp = df['dbp'].values
    subject_id = df['subjectID'].values
    
    # 4) 資料分割
    logger.info("\n4. 切分訓練集和測試集...")
    X_train, X_test, y_sbp_train, y_sbp_test, subid_train, subid_test = \
        train_test_split(X, y_sbp, subject_id, test_size=0.2, random_state=42)
    logger.info(f"訓練集大小: {len(X_train)}, 測試集大小: {len(X_test)}")
    
    # 5) 訓練評估
    logger.info("\n5. 訓練並評估多個模型...")
    results_sbp = train_and_eval_regressors(
        X_train, y_sbp_train,
        X_test, y_sbp_test,
        model_name="SBP"
    )
    
    # 6) 找出最佳模型
    best_model_name = min(results_sbp.keys(),
                         key=lambda k: results_sbp[k][0])
    best_model_sbp = results_sbp[best_model_name][2]
    logger.info(f"\n最佳SBP模型: {best_model_name}")
    
    # 7) 個人化校正示範
    logger.info("\n6. 個人化校正示範...")
    subjectA = list(set(subid_test))[0]  # 取第一個測試集subject做示範
    idxA = [i for i,v in enumerate(subid_test) if v==subjectA]
    
    if len(idxA) >= 2:
        half = len(idxA)//2
        calib_idxA = idxA[:half]
        eval_idxA = idxA[half:]
        X_calibA = X_test[calib_idxA]
        y_calibA = y_sbp_test[calib_idxA]
        X_evalA = X_test[eval_idxA]
        y_evalA = y_sbp_test[eval_idxA]
        
        offsetA = compute_personal_offset(best_model_sbp, X_calibA, y_calibA)
        preds_evalA_nocorr = best_model_sbp.predict(X_evalA)
        preds_evalA_withcorr = apply_offset(best_model_sbp, offsetA, X_evalA)
        
        mae_nocorr = mean_absolute_error(y_evalA, preds_evalA_nocorr)
        mae_withcorr = mean_absolute_error(y_evalA, preds_evalA_withcorr)
        logger.info(f"[Subject={subjectA}] 校正資料數={len(calib_idxA)}, 評估資料數={len(eval_idxA)}")
        logger.info(f"MAE(無個人化)={mae_nocorr:.3f}, MAE(有個人化)={mae_withcorr:.3f}")
    else:
        logger.warning(f"測試集中subject {subjectA}的資料不足，跳過校正")
    
    logger.info("\n=== 所有流程完成 ===")

def get_age_range_index(age):
    if age < 30:
        return 0
    elif age < 60:
        return 1
    else:
        return 2

def get_gender_index(gender):
    # 假設 0=male, 1=female，或反之
    if gender == 70:
        return 0
    else:
        return 1
    
def get_subset_index(age, gender):
    # subset_index = gender_index*3 + age_index
    # 0: male<30, 1: male30-60, 2: male>=60
    # 3: female<30,4: female30-60, 5: female>=60
    ar = get_age_range_index(age)
    gr = get_gender_index(gender)
    return gr * 3 + ar

def split_dataset_by_6groups(df):
    """
    將 DataFrame df 依照 (gender, age) 分為 6 個子集，回傳 List[DataFrame]，共 6 個。
    假設 df 裡至少有 'age' 與 'gender' 欄位。
    """
    from tqdm import tqdm
    
    groups_data = [[] for _ in range(6)]  # list of index-lists, each for one subgroup

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        age_val = row['age']
        gender_val = row['gender']
        subset_idx = get_subset_index(age_val, gender_val)
        groups_data[subset_idx].append(i)

    group_dfs = []
    for i in range(6):
        if len(groups_data[i])>0:
            subdf = df.iloc[groups_data[i]]
        else:
            subdf = pd.DataFrame(columns=df.columns)  # 空的
        group_dfs.append(subdf)
    return group_dfs


def main2():
    """
    執行 RandomForest 回歸，分別針對 SBP 和 DBP 在六個族群及全部資料進行訓練和評估。
    """
    logger.info("=== 血壓回歸模型訓練 (Random Forest) - main2 ===")
    
    # 1. 準備檔案路徑
    logger.info("\n1. 準備檔案路徑...")
    data_dir = Path('training_data_VitalDB_quality')
    train_files = [data_dir / f"training_{i+1}.h5" for i in range(9)]
    val_file = data_dir / 'validation.h5'
    test_file = data_dir / 'test.h5'
    
    # 檢查檔案是否存在
    existing_train_files = [f for f in train_files if f.exists()]
    if len(existing_train_files) != 9:
        logger.error(f"訓練檔案數量不足，找到 {len(existing_train_files)} 個，預期 9 個。")
        return
    
    if not val_file.exists() or not test_file.exists():
        logger.error("驗證檔案或測試檔案不存在")
        return
    
    # 2. 載入資料
    logger.info("\n2. 載入資料...")
    train_df = load_data_to_df(existing_train_files)
    val_df = load_data_to_df([val_file])
    test_df = load_data_to_df([test_file])
    
    # 3. 定義特徵和標籤
    logger.info("\n3. 定義特徵和標籤...")
    feature_cols = ['age', 'gender', 'weight', 'height', 'ptt', 'pat']
    target_cols = ['sbp', 'dbp']  # 分別預測 SBP 和 DBP
    
    # 4. 分群
    logger.info("\n4. 將資料分為六個族群...")
    train_groups = split_dataset_by_6groups(train_df)
    val_groups = split_dataset_by_6groups(val_df)
    test_groups = split_dataset_by_6groups(test_df)
    
    # 5. 分別對 SBP 和 DBP 進行訓練和評估
    for target_col in target_cols:
        logger.info(f"\n===== 開始訓練 {target_col.upper()} 模型 =====")
        results = {}
        
        # 5.1 訓練並評估每個族群的模型
        for i in range(6):
            logger.info(f"\n----- 訓練並評估 群組 {i} ({target_col.upper()}) -----")
            # 訓練集
            group_train = train_groups[i]
            X_train = group_train[feature_cols].values
            y_train = group_train[target_col].values
            
            # 驗證集
            group_val = val_groups[i]
            X_val = group_val[feature_cols].values
            y_val = group_val[target_col].values
            
            # 測試集
            group_test = test_groups[i]
            X_test = group_test[feature_cols].values
            y_test = group_test[target_col].values
            
            logger.info(f"群組 {i} - 訓練集大小: {len(X_train)}, 驗證集大小: {len(X_val)}, 測試集大小: {len(X_test)}")
            
            if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                logger.warning(f"群組 {i} 的某些資料集為空，跳過該群組。")
                continue
            
            # 訓練 RandomForest 模型
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # 評估 on Validation set
            y_val_pred = rf.predict(X_val)
            mae_val = mean_absolute_error(y_val, y_val_pred)
            r2_val = r2_score(y_val, y_val_pred)
            logger.info(f"群組 {i} - Validation: MAE={mae_val:.3f}, R²={r2_val:.3f}")
            
            # 評估 on Test set
            y_test_pred = rf.predict(X_test)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)
            logger.info(f"群組 {i} - Test: MAE={mae_test:.3f}, R²={r2_test:.3f}")
            
            # 儲存結果
            results[f"Group_{i}"] = {
                "Validation_MAE": mae_val,
                "Validation_R2": r2_val,
                "Test_MAE": mae_test,
                "Test_R2": r2_test
            }
        
        # 5.2 訓練並評估全體模型
        logger.info(f"\n----- 訓練並評估 全體模型 ({target_col.upper()}) -----")
        # 全體訓練集
        X_train_all = train_df[feature_cols].values
        y_train_all = train_df[target_col].values
        
        # 全體驗證集
        X_val_all = val_df[feature_cols].values
        y_val_all = val_df[target_col].values
        
        # 全體測試集
        X_test_all = test_df[feature_cols].values
        y_test_all = test_df[target_col].values
        
        logger.info(f"全體模型 - 訓練集大小: {len(X_train_all)}, 驗證集大小: {len(X_val_all)}, 測試集大小: {len(X_test_all)}")
        
        # 訓練全體 RandomForest
        rf_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_all.fit(X_train_all, y_train_all)
        
        # 評估 on Validation set
        y_val_pred_all = rf_all.predict(X_val_all)
        mae_val_all = mean_absolute_error(y_val_all, y_val_pred_all)
        r2_val_all = r2_score(y_val_all, y_val_pred_all)
        logger.info(f"全體模型 - Validation: MAE={mae_val_all:.3f}, R²={r2_val_all:.3f}")
        
        # 評估 on Test set
        y_test_pred_all = rf_all.predict(X_test_all)
        mae_test_all = mean_absolute_error(y_test_all, y_test_pred_all)
        r2_test_all = r2_score(y_test_all, y_test_pred_all)
        logger.info(f"全體模型 - Test: MAE={mae_test_all:.3f}, R²={r2_test_all:.3f}")
        
        # 儲存全體結果
        results["All_Groups"] = {
            "Validation_MAE": mae_val_all,
            "Validation_R2": r2_val_all,
            "Test_MAE": mae_test_all,
            "Test_R2": r2_test_all
        }
        
        # 6. 輸出該目標變數的所有結果
        logger.info(f"\n=== {target_col.upper()} 所有群組模型結果 ===")
        for group, metrics in results.items():
            logger.info(f"{group}:")
            logger.info(f"  Validation - MAE: {metrics['Validation_MAE']:.3f}, R²: {metrics['Validation_R2']:.3f}")
            logger.info(f"  Test       - MAE: {metrics['Test_MAE']:.3f}, R²: {metrics['Test_R2']:.3f}")
        
        # 7. 保存結果到 CSV
        results_df = pd.DataFrame(results).T
        timestamp = time()
        results_df.to_csv(f'random_forest_6groups_{target_col}_results_{int(timestamp)}.csv')
        logger.info(f"\n{target_col.upper()} 所有群組的結果已保存至 'random_forest_6groups_{target_col}_results_{int(timestamp)}.csv'")
    
    logger.info("\n=== main2 完成 ===")

if __name__ == '__main__':
    try:
        main2()
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}", exc_info=True)