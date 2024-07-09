import numpy as np
import pandas as pd
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from load_normap_ABP_data import load_training_set
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
from scipy import signal

from scipy.interpolate import interp1d

def preprocess_pulse(pulse, target_length=100): #target_length=200似乎就該選更高許多的threshold值
    # 计算插值点
    x_old = np.linspace(0, len(pulse) - 1, num=len(pulse))
    x_new = np.linspace(0, len(pulse) - 1, num=target_length)
    
    # 创建插值函数
    interp_func = interp1d(x_old, pulse, kind='linear')
    
    # 重采样到目标长度
    pulse_resampled = interp_func(x_new)
    
    # 去除线性趋势
    pulse_detrended = signal.detrend(pulse_resampled)
    
    # 均值中心化
    pulse_centered = pulse_detrended - np.mean(pulse_detrended)
    
    # 幅度归一化到[-1, 1]范围
    pulse_normalized = pulse_centered / np.max(np.abs(pulse_centered))
    
    # 端点对齐（可选）
    pulse_aligned = pulse_normalized - pulse_normalized[0]
    pulse_aligned[-1] = 0
    
    return pulse_aligned

# def preprocess_pulse(pulse, target_length=100):
#     # 重采样到目标长度
#     pulse_resampled = signal.resample(pulse, target_length)
    
#     # 去除线性趋势
#     pulse_detrended = signal.detrend(pulse_resampled)
    
#     # 均值中心化
#     pulse_centered = pulse_detrended - np.mean(pulse_detrended)
    
#     # 幅度归一化到[-1, 1]范围
#     pulse_normalized = pulse_centered / np.max(np.abs(pulse_centered))
    
#     # 端点对齐（可选）
#     pulse_aligned = pulse_normalized - pulse_normalized[0]
#     pulse_aligned[-1] = 0
    
#     return pulse_aligned


def calculate_dtw_scores(abp_data, turns):
    pulses = np.split(abp_data, turns)
    pulses = pulses[1:-1]
    processed_pulses = [preprocess_pulse(pulse) for pulse in pulses]
    dtw_distances = [dtw.distance(processed_pulses[i], processed_pulses[i+1]) for i in range(len(processed_pulses) - 1)]
    return dtw_distances

def preprocess_and_extract_features(abp_segments, abp_turns, abp_speaks):
    features = []
    for segment, turns, speaks in zip(abp_segments, abp_turns, abp_speaks):
        features_dict = {
            'mean': np.mean(segment),
            'std': np.std(segment),
            'min': np.min(segment),
            'max': np.max(segment),
            'median': np.median(segment),
            'mean_pulse_duration': np.mean(np.diff(turns)),
            'std_pulse_duration': np.std(np.diff(turns)),
            'num_pulses': len(turns) - 1,
            'mean_peak_value': np.mean(segment[speaks]),
            'std_peak_value': np.std(segment[speaks]),
        }
        features.append(features_dict)
    
    features_df = pd.DataFrame(features)
    
    tsfresh_features = extract_tsfresh_features(abp_segments)
    
    all_features = pd.concat([features_df, tsfresh_features], axis=1)
    
    return all_features

def extract_tsfresh_features(abp_segments):
    df = pd.DataFrame()
    for i, segment in enumerate(abp_segments):
        temp_df = pd.DataFrame({'id': i, 'time': range(len(segment)), 'value': segment})
        df = pd.concat([df, temp_df])
    
    features = extract_features(df, column_id='id', column_sort='time', column_value='value')
    
    return features

def handle_missing_values(features):
    features_with_missing = features.columns[features.isna().any()].tolist()
    features = features.drop(columns=features_with_missing)
    
    return features

def select_features(features, labels):
    selector = SelectKBest(f_classif, k=100)  # 選擇100個最佳特徵
    selector.fit(features, labels)
    selected_features = features.columns[selector.get_support()]
    
    return selected_features


def train_models(features, labels):
    selected_features = select_features(features, labels)
    print(f'selected_features : {selected_features}')
    X_train = features[selected_features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    svm.fit(X_train_scaled)
    
    if_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, random_state=42)
    if_model.fit(X_train_scaled)
    
    return svm, if_model, scaler, selected_features

def extract_features_for_segment(segment, turns, speaks):
    features_dict = {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'min': np.min(segment),
        'max': np.max(segment),
        'median': np.median(segment),
        'mean_pulse_duration': np.mean(np.diff(turns)),
        'std_pulse_duration': np.std(np.diff(turns)),
        'num_pulses': len(turns) - 1,
        'mean_peak_value': np.mean(segment[speaks]),
        'std_peak_value': np.std(segment[speaks]),
    }
    
    df = pd.DataFrame({'id': 0, 'time': range(len(segment)), 'value': segment})
    tsfresh_features = extract_features(df, column_id='id', column_sort='time', column_value='value')
    
    features = pd.concat([pd.DataFrame([features_dict]), tsfresh_features], axis=1)
    
    return features

def detect_anomaly(features, svm, if_model, scaler):
    features_scaled = scaler.transform(features)
    
    svm_score = -svm.decision_function(features_scaled)
    if_score = -if_model.decision_function(features_scaled)

    anomaly_score = (svm_score + if_score*5) / 2
    print(f'svm_score : {svm_score}, if_score : {if_score}, anomaly_score : {anomaly_score}')
    return anomaly_score[0]

def detect_anomaly_gui(segment, turns, speaks):
    with open('models.pkl', 'rb') as f:
        svm, if_model, scaler, selected_features, _, _, _ = pickle.load(f)
    
    features = extract_features_for_segment(segment, turns, speaks)
    features = features[selected_features]  # 只使用選擇的特徵
    anomaly_score = detect_anomaly(features, svm, if_model, scaler)
    
    return anomaly_score

def main():
    try:
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
        print("Loaded preprocessed features from file.")
    except FileNotFoundError:
        abp_segments, abp_turns, abp_speaks = load_training_set()
        print(f"Loaded {len(abp_segments)} segments, shape: {abp_segments.shape}")
        
        features = preprocess_and_extract_features(abp_segments, abp_turns, abp_speaks)
        print(f"Extracted {features.shape} features.")
        
        features = handle_missing_values(features)
        print("Handled missing values.")
        
        with open('features.pkl', 'wb') as f:
            pickle.dump(features, f)
        print("Saved preprocessed features to file.")
    
    labels = np.zeros(features.shape[0])  # 假設所有的訓練數據都是正常的
    svm, if_model, scaler, selected_features = train_models(features, labels)
    print(f"Models trained successfully with {len(selected_features)} selected features.")
    
    # 計算訓練集中每個樣本的異常分數
    X_train = features[selected_features]
    X_train_scaled = scaler.transform(X_train)
    svm_scores = -svm.decision_function(X_train_scaled)
    if_scores = -if_model.decision_function(X_train_scaled)
    anomaly_scores = (svm_scores + if_scores*5) / 2
    
    with open('models.pkl', 'wb') as f:
        pickle.dump((svm, if_model, scaler, selected_features, svm_scores, if_scores, anomaly_scores), f)
    print("Saved trained models and anomaly scores to file.")

if __name__ == '__main__':
    main()
    # import time
    # # 假設abp_data和turns已經存在
    # abp_data = np.random.randn(50000)  # 假設有一段ABP數據
    # turns = np.arange(0, 50000, 125)   # 假設每100個點有一個turn

    # def preprocess_pulse_resample(pulse, target_length=100):
    #     pulse_resampled = signal.resample(pulse, target_length)
    #     pulse_detrended = signal.detrend(pulse_resampled)
    #     pulse_centered = pulse_detrended - np.mean(pulse_detrended)
    #     pulse_normalized = pulse_centered / np.max(np.abs(pulse_centered))
    #     pulse_aligned = pulse_normalized - pulse_normalized[0]
    #     pulse_aligned[-1] = 0
    #     return pulse_aligned
    # def preprocess_pulse_interp2(pulse, target_length=200):
    #     x_old = np.linspace(0, len(pulse) - 1, num=len(pulse))
    #     x_new = np.linspace(0, len(pulse) - 1, num=target_length)
    #     interp_func = interp1d(x_old, pulse, kind='linear')
    #     pulse_resampled = interp_func(x_new)
    #     pulse_detrended = signal.detrend(pulse_resampled)
    #     pulse_centered = pulse_detrended - np.mean(pulse_detrended)
    #     pulse_normalized = pulse_centered / np.max(np.abs(pulse_centered))
    #     pulse_aligned = pulse_normalized - pulse_normalized[0]
    #     pulse_aligned[-1] = 0
    #     return pulse_aligned

    # def preprocess_pulse_interp(pulse, target_length=100):
    #     x_old = np.linspace(0, len(pulse) - 1, num=len(pulse))
    #     x_new = np.linspace(0, len(pulse) - 1, num=target_length)
    #     interp_func = interp1d(x_old, pulse, kind='linear')
    #     pulse_resampled = interp_func(x_new)
    #     pulse_detrended = signal.detrend(pulse_resampled)
    #     pulse_centered = pulse_detrended - np.mean(pulse_detrended)
    #     pulse_normalized = pulse_centered / np.max(np.abs(pulse_centered))
    #     pulse_aligned = pulse_normalized - pulse_normalized[0]
    #     pulse_aligned[-1] = 0
    #     return pulse_aligned

    # def calculate_dtw_scores(abp_data, turns, preprocess_func):
    #     pulses = np.split(abp_data, turns)
    #     dtw_distances = []
    #     pulses = pulses[1:-1]
    #     processed_pulses = [preprocess_func(pulse) for pulse in pulses]
    #     for i in range(len(processed_pulses) - 1):
    #         distance = dtw.distance(processed_pulses[i], processed_pulses[i+1])
    #         dtw_distances.append(distance)
    #     return dtw_distances

    # # 測量使用resample方法的時間
    # start_time_resample = time.time()
    # dtw_scores_resample = calculate_dtw_scores(abp_data, turns, preprocess_pulse_interp2)#preprocess_pulse_resample)
    # end_time_resample = time.time()
    # time_resample = end_time_resample - start_time_resample

    # # 測量使用interp方法的時間
    # start_time_interp = time.time()
    # dtw_scores_interp = calculate_dtw_scores(abp_data, turns, preprocess_pulse_interp)
    # end_time_interp = time.time()
    # time_interp = end_time_interp - start_time_interp

    # print(f'diff of two methods: {np.array(dtw_scores_resample) - np.array(dtw_scores_interp)}')
    # print(f'Interpolate2 method time: {time_resample:.6f} seconds')
    # print(f'Interpolate method time: {time_interp:.6f} seconds')