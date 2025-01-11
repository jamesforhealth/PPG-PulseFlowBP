##################################################
# bp_vascular_regression.py
##################################################
import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

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

        # numpy return
        return pers, vasc, label


def load_basic_data_for_sklearn(dataset):
    """
    將 dataset 轉成 X=(N,6), y=(N,2) for sklearn
    """
    X_list=[]
    y_list=[]
    for i in range(len(dataset)):
        pers, vasc, lab = dataset[i]  # pers(4,), vasc(2,), lab(2,)
        X_feat= np.concatenate([pers, vasc], axis=0)  # =>(6,)
        X_list.append(X_feat)
        y_list.append(lab)
    X_arr= np.array(X_list)  # (N,6)
    y_arr= np.array(y_list)  # (N,2)
    return X_arr, y_arr


############################
# (B) scikit-learn training
############################
def train_random_forest(X_train, y_train, X_val, y_val):
    rf= RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    preds= rf.predict(X_val)  # (val_size,2)
    mae_sbp= np.mean(np.abs(preds[:,0]- y_val[:,0]))
    mae_dbp= np.mean(np.abs(preds[:,1]- y_val[:,1]))
    mae= (mae_sbp+ mae_dbp)/2
    print(f"[RandomForest] val MAE= {mae:.4f} (SBP={mae_sbp:.4f}, DBP={mae_dbp:.4f})")
    return rf

def train_svm(X_train, y_train, X_val, y_val):
    # SVR 不支援多輸出 => 用MultiOutputRegressor包裝
    base_svr = SVR(kernel='rbf', C=10.0, max_iter=10000)
    multi_svr= MultiOutputRegressor(base_svr)
    multi_svr.fit(X_train, y_train)

    preds= multi_svr.predict(X_val)
    mae_sbp= np.mean(np.abs(preds[:,0]- y_val[:,0]))
    mae_dbp= np.mean(np.abs(preds[:,1]- y_val[:,1]))
    mae= (mae_sbp+ mae_dbp)/2
    print(f"[SVM] val MAE= {mae:.4f} (SBP={mae_sbp:.4f}, DBP={mae_dbp:.4f})")
    return multi_svr


############################
# (C) main
############################
if __name__=='__main__':
    data_dir= Path('training_data_VitalDB')
    train_files= [ data_dir/f"training_{i+1}.h5" for i in range(9) ]
    val_file  = data_dir/'validation.h5'

    # 1) 建立 Dataset
    from torch.utils.data import ConcatDataset
    train_dss=[]
    for tf in train_files:
        if tf.exists():
            train_dss.append(VitalSignDatasetBasic(str(tf)))
    train_dataset= ConcatDataset(train_dss)
    val_dataset= VitalSignDatasetBasic(str(val_file))

    # 2) 轉成 X,y
    X_train, y_train= load_basic_data_for_sklearn(train_dataset)
    X_val, y_val   = load_basic_data_for_sklearn(val_dataset)

    # 3) RandomForest
    rf_model= train_random_forest(X_train, y_train, X_val, y_val)

    # 4) SVM
    # svm_model= train_svm(X_train, y_train, X_val, y_val)

    # 如需測試/預測:
    # test_files= data_dir/'test.h5'
    # test_dataset= VitalSignDatasetBasic(str(test_files))
    # X_test, y_test= load_basic_data_for_sklearn(test_dataset)
    # preds_rf= rf_model.predict(X_test)
    # ...
