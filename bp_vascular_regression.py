import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

############################
# (A) Dataset
############################
class VitalSignDatasetBasic(Dataset):
    """
    讀取 .h5，只取:
      personal_info: (N,4)
      vascular_properties: (N,2)
      segsbp, segdbp => label=(N,2)
    不需要 ppg/annotations。
    """
    def __init__(self, h5_file):
        super().__init__()
        self.h5 = h5py.File(h5_file, 'r')
        self.personal = self.h5['personal_info']        # (N,4)
        self.vascular = self.h5['vascular_properties']  # (N,2)
        sbp = self.h5['segsbp'][:]                      # (N,)
        dbp = self.h5['segdbp'][:]
        self.labels = np.stack([sbp, dbp], axis=1)      # (N,2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pers = self.personal[idx]       # shape=(4,)
        vasc = self.vascular[idx]       # shape=(2,)
        label= self.labels[idx]         # shape=(2,)
        return pers, vasc, label

def load_basic_data_for_sklearn(dataset):
    """
    將 dataset 轉成 X=(N,6), y=(N,2) for sklearn
    """
    X_list = []
    y_list = []
    pers_list = []  # 分開存儲個人特徵
    vasc_list = []  # 分開存儲血管特徵
    
    for i in range(len(dataset)):
        pers, vasc, lab = dataset[i]
        pers_list.append(pers)
        vasc_list.append(vasc)
        X_feat = np.concatenate([pers, vasc], axis=0)  # 合併所有特徵
        X_list.append(X_feat)
        y_list.append(lab)
    
    X_arr = np.array(X_list)      # (N,6)
    y_arr = np.array(y_list)      # (N,2)
    pers_arr = np.array(pers_list)  # (N,4)
    vasc_arr = np.array(vasc_list)  # (N,2)
    
    return X_arr, y_arr, pers_arr, vasc_arr

############################
# (B) Model Training & Analysis
############################
'''
def train_random_forest(X_train, y_train, X_val, y_val):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_val)
    mae_sbp = np.mean(np.abs(preds[:,0] - y_val[:,0]))
    mae_dbp = np.mean(np.abs(preds[:,1] - y_val[:,1]))
    mae = (mae_sbp + mae_dbp)/2
    print(f"[RandomForest] val MAE= {mae:.4f} (SBP={mae_sbp:.4f}, DBP={mae_dbp:.4f})")
    return rf
'''
def analyze_feature_importance(rf_model, X_val, y_val, feature_names):
    """分析特徵重要性"""
    plt.figure(figsize=(12, 6))
    
    # 1. Random Forest 內建特徵重要性
    importance = rf_model.feature_importances_
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, importance)
    plt.title('Random Forest Feature Importance')
    plt.xticks(rotation=45)
    
    # 2. Permutation Importance
    result = permutation_importance(rf_model, X_val, y_val, n_repeats=10)
    perm_importance = result.importances_mean
    plt.subplot(1, 2, 2)
    plt.bar(feature_names, perm_importance)
    plt.title('Permutation Feature Importance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return importance, perm_importance

def analyze_feature_correlation(X, feature_names):
    """分析特徵相關性"""
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X.T)
    sns.heatmap(corr_matrix, 
                annot=True, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def analyze_all_feature_combinations(train_dataset, val_dataset):
    """分析不同特徵組合的效果"""
    X_train, y_train, pers_train, vasc_train = load_basic_data_for_sklearn(train_dataset)
    X_val, y_val, pers_val, vasc_val = load_basic_data_for_sklearn(val_dataset)
    
    # 1. 使用所有特徵
    print("\nTraining with ALL features:")
    rf_all = train_random_forest(X_train, y_train, X_val, y_val)
    
    # 2. 只使用個人特徵
    print("\nTraining with PERSONAL features only:")
    rf_pers = train_random_forest(pers_train, y_train, pers_val, y_val)
    
    # 3. 只使用血管特徵
    print("\nTraining with VASCULAR features only:")
    rf_vasc = train_random_forest(vasc_train, y_train, vasc_val, y_val)
    
    return rf_all, X_val, y_val

############################
# (C) main
############################

from xgboost import XGBRegressor

def train_random_forest(X_train, y_train, X_val, y_val, max_depth=None):
    """增加max_depth參數，設為None時不限制深度"""
    rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_val)
    mae_sbp = np.mean(np.abs(preds[:,0] - y_val[:,0]))
    mae_dbp = np.mean(np.abs(preds[:,1] - y_val[:,1]))
    mae = (mae_sbp + mae_dbp)/2
    print(f"[RandomForest] val MAE= {mae:.4f} (SBP={mae_sbp:.4f}, DBP={mae_dbp:.4f})")
    return rf

def train_xgboost(X_train, y_train, X_val, y_val):
    """使用XGBoost進行訓練"""
    # 使用MultiOutputRegressor包裝XGBoost以處理多輸出
    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
    )
    xgb.fit(X_train, y_train)

    preds = xgb.predict(X_val)
    mae_sbp = np.mean(np.abs(preds[:,0] - y_val[:,0]))
    mae_dbp = np.mean(np.abs(preds[:,1] - y_val[:,1]))
    mae = (mae_sbp + mae_dbp)/2
    print(f"[XGBoost] val MAE= {mae:.4f} (SBP={mae_sbp:.4f}, DBP={mae_dbp:.4f})")
    return xgb

def analyze_single_features(train_dataset, val_dataset, feature_names):
    """分析單一特徵的預測效果"""
    X_train, y_train, pers_train, vasc_train = load_basic_data_for_sklearn(train_dataset)
    X_val, y_val, pers_val, vasc_val = load_basic_data_for_sklearn(val_dataset)
    
    results = []
    print("\nSingle Feature Analysis:")
    for i in range(X_train.shape[1]):
        X_train_single = X_train[:, i:i+1]  # 保持2D形狀
        X_val_single = X_val[:, i:i+1]
        
        # 使用不限深度的RandomForest
        rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
        rf.fit(X_train_single, y_train)
        
        preds = rf.predict(X_val_single)
        mae_sbp = np.mean(np.abs(preds[:,0] - y_val[:,0]))
        mae_dbp = np.mean(np.abs(preds[:,1] - y_val[:,1]))
        mae = (mae_sbp + mae_dbp)/2
        
        results.append({
            'feature': feature_names[i],
            'mae': mae,
            'mae_sbp': mae_sbp,
            'mae_dbp': mae_dbp
        })
        print(f"Feature '{feature_names[i]}': MAE= {mae:.4f} (SBP={mae_sbp:.4f}, DBP={mae_dbp:.4f})")
    
    # 繪製單特徵預測結果比較
    plt.figure(figsize=(12, 6))
    features = [r['feature'] for r in results]
    maes = [r['mae'] for r in results]
    plt.bar(features, maes)
    plt.title('Single Feature Prediction Performance')
    plt.xlabel('Features')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_models_and_depths(train_dataset, val_dataset):
    """分析不同模型和深度的效果"""
    X_train, y_train, pers_train, vasc_train = load_basic_data_for_sklearn(train_dataset)
    X_val, y_val, pers_val, vasc_val = load_basic_data_for_sklearn(val_dataset)
    
    print("\nTraining RandomForest with different depths:")
    # 1. RandomForest with max_depth=10 (original)
    print("\nRandomForest (max_depth=10):")
    rf_10 = train_random_forest(X_train, y_train, X_val, y_val, max_depth=10)
    
    # 2. RandomForest with max_depth=20
    print("\nRandomForest (max_depth=20):")
    rf_20 = train_random_forest(X_train, y_train, X_val, y_val, max_depth=20)
    
    # 3. RandomForest with unlimited depth
    print("\nRandomForest (unlimited depth):")
    rf_unlimited = train_random_forest(X_train, y_train, X_val, y_val, max_depth=None)
    
    # 4. XGBoost
    print("\nXGBoost:")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    return rf_unlimited, X_val, y_val

if __name__=='__main__':
    # 設定特徵名稱
    feature_names = ['Age', 'Gender', 'Height', 'Weight', 'PTT', 'PAT']
    
    # 載入數據
    data_dir = Path('training_data_VitalDB_quality')
    train_files = [data_dir/f"training_{i+1}.h5" for i in range(2)]
    val_file = data_dir/'validation.h5'

    # 建立 Dataset
    train_dss = []
    for tf in train_files:
        if tf.exists():
            train_dss.append(VitalSignDatasetBasic(str(tf)))
    train_dataset = ConcatDataset(train_dss)
    val_dataset = VitalSignDatasetBasic(str(val_file))

    # 分析不同模型和深度
    rf_model, X_val, y_val = analyze_models_and_depths(train_dataset, val_dataset)
    
    # 分析單一特徵的預測效果
    single_feature_results = analyze_single_features(train_dataset, val_dataset, feature_names)
    
    # 特徵重要性和相關性分析
    print("\nAnalyzing feature importance and correlations...")
    importance, perm_importance = analyze_feature_importance(rf_model, X_val, y_val, feature_names)
    
    X_all, _, _, _ = load_basic_data_for_sklearn(val_dataset)
    corr_matrix = analyze_feature_correlation(X_all, feature_names)