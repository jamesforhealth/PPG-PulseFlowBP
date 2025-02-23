import sys
import os
import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QCheckBox, QComboBox, QSlider, QLabel, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
# from model_abp_1250points import ConvAutoencoder, ConvAutoencoder2, predict_reconstructed_1250abp
# from model_pulse_representation import predict_reconstructed_abp
# from abp_anomaly_detection_algorithms import detect_anomaly_gui, calculate_dtw_scores
from load_normap_ABP_data import load_training_set
import pickle
import torch
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from prepare_training_data_vitaldb import check_ppg_quality
from ECG_segment_autoencoder import ECGAutoencoder
import torch.nn.functional as F
def lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.autoRange()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton:
            pg.ViewBox.mouseDragEvent(self, ev, axis)

    def wheelEvent(self, ev, axis=None):
        # 根據自己想要的行為來調整
        # 如果完全不需要用到 axis，可忽略之或傳給 super
        if axis is not None:
            # pyqtgraph 可能傳進來 axis=0/1，代表 x 或 y 軸
            pass
        if ev.modifiers() & Qt.ControlModifier:
            self.scaleBy((1.1**(-ev.delta() * 0.001), 1), center=(0.5, 0.5))
        else:
            self.scaleBy((1, 1.1**(-ev.delta() * 0.001)), center=(0.5, 0.5))

class PulseDBViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.ecg_model = ECGAutoencoder().to(self.device)
        # self.ecg_model.load_state_dict(torch.load('ecg_autoencoder_test.pth'))
        # self.ecg_model.eval()


        self.setWindowTitle("PulseDB Signal Viewer")    
        self.resize(1200, 800)

        self.training_abp_data = None
        self.training_abp_turns = None
        self.training_abp_speaks = None
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = ConvAutoencoder2().to(self.device)
        # self.model.load_state_dict(torch.load('abp1250model2.pt'))
        # self.model.eval()
        self.reconstructed_error = [0]

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 上部佈局
        top_layout = QHBoxLayout()

        # 左側面板 - 文件列表和控制
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.db_combo = QComboBox()
        self.db_combo.addItems(["PulseDB_Vital", "PulseDB_MIMIC", "training_set.npz"])
        self.db_combo.currentIndexChanged.connect(self.load_files)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_file)

        self.calc_button = QPushButton("Calculate Anomaly Score")
        # self.calc_button.clicked.connect(self.calculate_anomaly_score)
        left_layout.addWidget(self.calc_button)

        left_layout.addWidget(self.db_combo)
        left_layout.addWidget(self.file_list)
        self.anomaly_score_label = QLabel("Anomaly Score: N/A")
        left_layout.addWidget(self.anomaly_score_label)

        # self.svm_score_label = QLabel("SVM Score: N/A")
        # self.if_score_label = QLabel("IF Score: N/A")
        # left_layout.addWidget(self.svm_score_label)
        # left_layout.addWidget(self.if_score_label)
        # 複選框
        self.checkbox_smoothed = QCheckBox("Smoothed ABP")
        self.checkbox_smoothed.setChecked(False)
        self.checkbox_smoothed.stateChanged.connect(self.update_plot)
        left_layout.addWidget(self.checkbox_smoothed)

        # 添加二阶导数的复选框
        self.checkbox_second_derivative = QCheckBox("ABP Second Derivative")
        self.checkbox_second_derivative.setChecked(False)
        self.checkbox_second_derivative.stateChanged.connect(self.update_plot)
        left_layout.addWidget(self.checkbox_second_derivative)

        self.checkbox_double_integrated = QCheckBox("ABP Second Derivative")
        self.checkbox_double_integrated.setChecked(False)
        self.checkbox_double_integrated.stateChanged.connect(self.update_plot)
        left_layout.addWidget(self.checkbox_double_integrated)

        self.checkboxes = {}
        for signal in ['ABP_Raw', 'ABP_F', 'ABP_SPeaks', 'ABP_Turns', 'ECG_Raw', 'ECG_F', 'ECG_RPeaks', 'PPG_Raw', 'PPG_F', 'PPG_SPeaks', 'PPG_Turns', 'Reconstructed_ABP']:
            self.checkboxes[signal] = QCheckBox(signal)
            self.checkboxes[signal].setChecked(False) if signal in ['ABP_F', 'ECG_Raw', 'PPG_Raw'] else self.checkboxes[signal].setChecked(True)
            self.checkboxes[signal].stateChanged.connect(self.update_plot)
            left_layout.addWidget(self.checkboxes[signal])

        # 元信息顯示區域
        self.meta_info = QTextEdit()
        self.meta_info.setReadOnly(True)
        left_layout.addWidget(self.meta_info)

        # 在左下角添加显示区域
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        # 在左下方添加 segment 分析結果區域
        self.analysis_group = QGroupBox("Segment 分析結果")
        analysis_layout = QVBoxLayout()

        # 創建用於顯示分析結果的文本框
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(150)
        self.analysis_text.setMaximumHeight(200)
        
        # 設置字體
        font = QFont("Consolas", 9)
        self.analysis_text.setFont(font)
        
        # 設置樣式
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 5px;
            }
        """)
        
        analysis_layout.addWidget(self.analysis_text)
        self.analysis_group.setLayout(analysis_layout)
        
        # 添加到左側布局
        left_layout.addWidget(self.analysis_group)

        # # 右側面板 - 繪圖區域
        # self.plot_widget = pg.PlotWidget(viewBox=CustomViewBox())
        # self.plot_widget.setBackground('w')
        # self.plot_widget.setLabel('left', 'Amplitude')
        # self.plot_widget.setLabel('bottom', 'Sample Points')
        # self.plot_widget.setMenuEnabled(False)
        # self.plot_widget.setClipToView(True)
        # self.plot_widget.showGrid(x=True, y=True)
 # 右側面板 - 繪圖區域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 創建兩個 PlotWidget
        self.plot_widget = pg.PlotWidget(viewBox=CustomViewBox())
        self.normalized_plot_widget = pg.PlotWidget(viewBox=CustomViewBox())

        for widget in [self.plot_widget, self.normalized_plot_widget]:
            widget.setBackground('w')
            widget.setLabel('left', 'Amplitude')
            widget.setLabel('bottom', 'Sample Points')
            widget.setMenuEnabled(False)
            widget.setClipToView(True)
            widget.showGrid(x=True, y=True)

        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.normalized_plot_widget)

        # 添加左側和右側面板到上部佈局
        top_layout.addWidget(left_panel, 1)
        top_layout.addWidget(right_panel, 7)

        self.main_layout.addLayout(top_layout)

        self.reconstruction_error_plot = pg.PlotWidget()
        self.reconstruction_error_plot.setBackground('w')
        self.reconstruction_error_plot.setLabel('left', 'Signal Error')
        self.reconstruction_error_plot.setLabel('bottom', 'Time')
        self.main_layout.addWidget(self.reconstruction_error_plot)


        # 底部滑動條
        self.segment_slider = QSlider(Qt.Horizontal)
        self.segment_slider.setMinimum(0)
        self.segment_slider.valueChanged.connect(self.update_plot)
        self.segment_label = QLabel("Segment: 0")

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.segment_label)
        bottom_layout.addWidget(self.segment_slider)
        
        # 添加一个标签用于显示信号质量检查结果
        self.quality_label = QLabel("信号质量：未知")
        bottom_layout.addWidget(self.quality_label)  # 将其添加到底部布局中

        self.main_layout.addLayout(bottom_layout)

        self.data = {}
        self.current_file_path = ""

        self.load_files()

        self.analyze_peaks_button = QPushButton("分析Peaks對齊情況")
        self.analyze_peaks_button.clicked.connect(self.analyze_peaks_alignment)
        left_layout.addWidget(self.analyze_peaks_button)

        # 定义颜色
        self.colors = {
            'ABP': (255, 0, 0),          # 红色
            'ABP_SPeaks': (0, 255, 0),   # 绿色
            'ABP_Turns': (0, 0, 255),    # 蓝色
            # ... 其他颜色定义
        }

    def load_files(self):
        selected_db = self.db_combo.currentText()
        if selected_db == "training_set.npz":
            self.file_list.clear()
            self.file_list.addItem(selected_db)
            for checkbox in self.checkboxes.values():
                checkbox.setVisible(False)
        else:
            folder_path = os.path.join("D:\\PulseDB", selected_db)
            self.file_list.clear()
            for file in os.listdir(folder_path):
                if file.endswith(".mat"):
                    self.file_list.addItem(file)
            for checkbox in self.checkboxes.values():
                checkbox.setVisible(True)

    def load_selected_file(self, item):
        selected_db = self.db_combo.currentText()
        if selected_db == "training_set.npz":
            self.load_training_set_info(selected_db)
        else:
            db_folder = self.db_combo.currentText()
            self.current_file_path = os.path.join("D:\\PulseDB", db_folder, item.text())
            self.load_mat_file(self.current_file_path)
        self.update_plot()

    def load_mat_file(self, file_path):
        print(f"Loading file: {file_path}")
        with h5py.File(file_path, 'r') as f:
            matdata = f['Subj_Wins']
            print(f"Available keys: {list(matdata.keys())}")
            self.data = {}
            
            for key in matdata.keys():
                print(f"\nProcessing key: {key}")
                try:
                    if key in ['ABP_Raw', 'ABP_F', 'ECG_Raw', 'ECG_F', 'PPG_Raw', 'PPG_F']:
                        self.data[key] = []
                        for i, ref in enumerate(matdata[key][0]):
                            data = f[ref][:]
                            # print(f"  {key} segment {i}: shape={data.shape}, dtype={data.dtype}")
                            self.data[key].append(data.flatten())
                            
                    elif key in ['ABP_SPeaks', 'ABP_Turns', 'ECG_RPeaks', 'ECG_RealPeaks', 'PPG_SPeaks', 'PPG_Turns']:
                        self.data[key] = []
                        for i, ref in enumerate(matdata[key][0]):
                            data = f[ref][:]
                            # print(f"  {key} segment {i}: {len(data)} points")
                            self.data[key].append(data.flatten().astype(np.int64))
                            
                    else:
                        self.data[key] = []
                        for i, ref in enumerate(matdata[key][0]):
                            value = f[ref][()]
                            if key == 'CaseID' or key == 'WinID' or key == 'WinSeqID':
                                print(f"  {key} value {i}: {value}")
                            if isinstance(value, np.ndarray):
                                value = value.squeeze()
                                if value.size == 1:
                                    value = value.item()
                                else:
                                    value = value.tolist()
                            self.data[key].append(value)
                            
                    # print(f"Successfully processed {key}")
                    
                except Exception as e:
                    print(f"Error processing {key}: {str(e)}")
                    self.data[key] = []

            self.segment_slider.setMaximum(len(self.data['ABP_Raw']) - 1)
            self.update_meta_info()
            self.update_plot()

    def load_training_set_info(self, file_path):
        self.training_abp_data, self.training_abp_turns, self.training_abp_speaks = load_training_set(file_path)
        self.segment_slider.setMaximum(len(self.training_abp_data) - 1)
        self.meta_info.setText(f"Total segments: {len(self.training_abp_data)}")
        # 載入預先計算好的異常分數
        with open('models.pkl', 'rb') as f:
            _, _, _, _, self.svm_scores, self.if_scores, self.anomaly_scores = pickle.load(f)


    def update_meta_info(self):
        if self.db_combo.currentText() == "training_set.npz":
            return
        meta_text = ""
        for key in ['Age', 'BMI', 'CaseID', 'Gender', 'Height', 'IncludeFlag',
                    'SegDBP', 'SegSBP', 'SegmentID', 'SubjectID', 'Weight',
                    'WinID', 'WinSeqID']:
            if key in self.data:
                value = self.data[key][0]  # 先取出第一个元素
                # print(f"Key: {key}, Value: {value}, Type: {type(value)}")  # 调试信息
# 检查 value 是否为可索引的（如数组）
                if isinstance(value, (np.ndarray, list)):
                    if len(value) == 1:
                        # 如果只有一个元素，取出该元素
                        value = value[0]
                    else:
                        # 如果有多个元素，将其转换为列表
                        value = value.tolist() if isinstance(value, np.ndarray) else value
                # 如果 value 是标量，直接使用
                meta_text += f"{key}: {value}\n"
        self.meta_info.setText(meta_text)

    def update_dtw_plot(self, dtw_scores, turns):
        self.reconstruction_error_plot.clear()
        x = turns[1:-1]
        y = dtw_scores
        # 绘制DTW曲线
        self.reconstruction_error_plot.plot(x, y, pen=pg.mkPen(color=(255, 0, 0), width=2))
        
        # 添加散点图以突出显示每个DTW值
        scatter = pg.ScatterPlotItem(x, y, size=10, brush=pg.mkBrush(255, 0, 0, 120))
        self.reconstruction_error_plot.addItem(scatter)
        
        self.reconstruction_error_plot.setLabel('left', 'DTW Distance')
        self.reconstruction_error_plot.setLabel('bottom', 'Sample Points')
        self.reconstruction_error_plot.setTitle('DTW Distances Between Adjacent Pulses')

        # 设置x轴范围为0到1250（10秒的采样点数）
        self.reconstruction_error_plot.setXRange(0, 1250)

        # 设置y轴范围，留出边距
        y_min, y_max = min(y), max(y)
        y_range = y_max - y_min
        self.reconstruction_error_plot.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)



    def compute_ecg_dtw_score(self, segment_idx):
        """
        1. 取得該 segment 的 ECG_F 資料 (長度 1250)
        2. 取得該 segment 的 R-peak 位置清單(ECG_RPeaks)
        3. 逐對相鄰peak切出 beat 片段
        4. 逐對相鄰 beat 做 DTW，取得距離
        5. 以 平均DTW距離 作為該 segment 的異常分數
           → 值越大，表示各心拍形狀差異越大(可能越異常)
        """
        fs = 125
        if 'ECG_F' not in self.data or 'ECG_RPeaks' not in self.data:
            return None
        
        ecg_f = self.data['ECG_F'][segment_idx]  # shape (1250,)
        peaks = self.data['ECG_RPeaks'][segment_idx]
        peaks = peaks[peaks < len(ecg_f)]  # 避免越界
        
        if len(peaks) < 3:
            # 若 R-peak 太少，無法形成有效 beat, 回傳無限大
            return float('inf')
        
        # 收集 beat 片段
        beat_segments = []
        for i in range(len(peaks)-1):
            start_idx = peaks[i]
            end_idx   = peaks[i+1]
            if end_idx <= start_idx:
                continue
            beat = ecg_f[start_idx:end_idx]
            beat_segments.append(beat)
        
        # 逐對相鄰 beat 計算 DTW
        distances = []
        for i in range(len(beat_segments)-1):
            dist = self.simple_dtw(beat_segments[i], beat_segments[i+1])
            distances.append(dist)
        
        if len(distances) == 0:
            return 0.0
        
        # 以平均 DTW 距離當異常分數
        anomaly_score = float(np.mean(distances))
        return anomaly_score

    def simple_dtw(self, seq1, seq2):
        """
        一個簡單的 DTW 實作；如需更快可考慮 fastdtw 或其他庫
        這裡僅示範概念
        """
        n, m = len(seq1), len(seq2)
        dtw_mat = np.full((n+1, m+1), np.inf)
        dtw_mat[0,0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_mat[i,j] = cost + min(dtw_mat[i-1,j],    # 上
                                          dtw_mat[i,j-1],    # 左
                                          dtw_mat[i-1,j-1])  # 斜上
        return dtw_mat[n,m]

    def update_plot(self):
        self.plot_widget.clear()
        self.normalized_plot_widget.clear()
        segment = self.segment_slider.value()
        self.segment_label.setText(f"Segment: {segment}")

        if self.db_combo.currentText() == "training_set.npz":
            if self.training_abp_data is not None:
                x = np.arange(1250)
                y = self.training_abp_data[segment]
                self.plot_widget.plot(x, y, pen=pg.mkPen(color=(255, 0, 0), width=2))
        else:
            if not self.data:
                return
            x = np.arange(1250)
            colors = {
                'ABP': (255, 0, 0),
                'ABP_SPeaks': (0, 255, 0),
                'ABP_Turns': (0, 0, 255),
                'ECG': (0, 255, 0),
                'PPG': (0, 0, 255),
                'ECG_peaks': (255, 0, 255),
                'ECG_real_peaks': (255, 255, 0),
                'PPG_peaks': (255, 165, 0),
                'PPG_turns': (0, 255, 255)
            }

            # ABP_Raw
            if self.checkboxes['ABP_Raw'].isChecked():
                if 'ABP_Raw' in self.data:
                    y_raw = self.data['ABP_Raw'][segment]
                    self.plot_widget.plot(x, y_raw, pen=pg.mkPen(color=colors['ABP'], width=2, name='ABP_Raw'))

            # ABP_F
            if self.checkboxes['ABP_F'].isChecked():
                if 'ABP_F' in self.data:
                    y_norm = self.data['ABP_F'][segment]
                    self.normalized_plot_widget.plot(x, y_norm, 
                        pen=pg.mkPen(color=colors['ABP'], width=2, name='ABP_F'))
            
            # ECG_Raw
            if self.checkboxes['ECG_Raw'].isChecked() and 'ECG_Raw' in self.data:
                y_ecg = self.data['ECG_Raw'][segment]
                self.plot_widget.plot(x, y_ecg, pen=pg.mkPen(color=colors['ECG'], width=2, name='ECG_Raw'))
                if self.checkboxes['ECG_RPeaks'].isChecked() and 'ECG_RPeaks' in self.data:
                    peaks = self.data['ECG_RPeaks'][segment]
                    peaks = peaks[peaks < 1250]
                    y_peaks = y_ecg[peaks]
                    self.plot_widget.plot(peaks, y_peaks, pen=None, 
                        symbol='o', symbolPen=None, symbolSize=4, symbolBrush=colors['ECG_peaks'])

            # ECG_F
            if self.checkboxes['ECG_F'].isChecked() and 'ECG_F' in self.data:
                ecg_f = self.data['ECG_F'][segment]
                self.normalized_plot_widget.plot(x, ecg_f, 
                    pen=pg.mkPen(color=colors['ECG'], width=2, name='ECG_F'))

                if self.checkboxes['ECG_RPeaks'].isChecked() and 'ECG_RPeaks' in self.data:
                    peaks = self.data['ECG_RPeaks'][segment]
                    peaks = peaks[peaks < 1250]
                    y_peaks = ecg_f[peaks]
                    scatter_peaks = pg.ScatterPlotItem(peaks, y_peaks,
                        symbol='o', size=6, pen=pg.mkPen(colors['ECG_peaks'], width=2),
                        brush=pg.mkBrush(colors['ECG_peaks']), name='ECG_RPeaks')
                    self.normalized_plot_widget.addItem(scatter_peaks)

            # PPG_Raw
            if self.checkboxes['PPG_Raw'].isChecked() and 'PPG_Raw' in self.data:
                y_ppg = self.data['PPG_Raw'][segment]
                self.plot_widget.plot(x, y_ppg, pen=pg.mkPen(color=colors['PPG'], width=2, name='PPG_Raw'))
                if self.checkboxes['PPG_SPeaks'].isChecked() and 'PPG_SPeaks' in self.data:
                    peaks = self.data['PPG_SPeaks'][segment]
                    peaks = peaks[peaks < 1250]
                    y_peaks = y_ppg[peaks]
                    self.plot_widget.plot(peaks, y_peaks, pen=None, 
                        symbol='o', symbolPen=None, symbolSize=4, symbolBrush=colors['PPG_peaks'])
                if self.checkboxes['PPG_Turns'].isChecked() and 'PPG_Turns' in self.data:
                    turns = self.data['PPG_Turns'][segment]
                    turns = turns[turns < 1250]
                    y_turns = y_ppg[turns]
                    self.plot_widget.plot(turns, y_turns, pen=None, 
                        symbol='o', symbolPen=None, symbolSize=4, symbolBrush=colors['PPG_turns'])
            
            # PPG_F
            if self.checkboxes['PPG_F'].isChecked() and 'PPG_F' in self.data:
                ppg_f = self.data['PPG_F'][segment]
                self.normalized_plot_widget.plot(x, ppg_f, 
                    pen=pg.mkPen(color=colors['PPG'], width=2, name='PPG_F'))
                
                if self.checkboxes['PPG_SPeaks'].isChecked() and 'PPG_SPeaks' in self.data:
                    peaks = self.data['PPG_SPeaks'][segment]
                    peaks = peaks[peaks < 1250]
                    y_peaks = ppg_f[peaks]
                    scatter_peaks = pg.ScatterPlotItem(peaks, y_peaks,
                        symbol='o', size=6, pen=pg.mkPen(colors['PPG_peaks'], width=2),
                        brush=pg.mkBrush(colors['PPG_peaks']), name='PPG_SPeaks')
                    self.normalized_plot_widget.addItem(scatter_peaks)

                if self.checkboxes['PPG_Turns'].isChecked() and 'PPG_Turns' in self.data:
                    turns = self.data['PPG_Turns'][segment]
                    turns = turns[turns < 1250]
                    y_turns = ppg_f[turns]
                    self.normalized_plot_widget.plot(turns, y_turns, pen=None, 
                        symbol='o', symbolPen=None, symbolSize=4, symbolBrush=colors['PPG_turns'])

        segment = self.segment_slider.value()
        fs = 125.0

        # 顯示信號質量 (以 PPG/ECG peaks 做簡單判斷)
        peaks_dict = {
            'ECG_RealPeaks': self.data.get('ECG_RPeaks', [])[segment],
            'PPG_SPeaks': self.data.get('PPG_SPeaks', [])[segment],
            'PPG_Turns': self.data.get('PPG_Turns', [])[segment],
        }
        if any(len(p) == 0 for p in peaks_dict.values()):
            self.quality_label.setText("信号质量：不合格（缺少特征点）")
        else:
            quality_pass = check_ppg_quality(peaks_dict)
            self.quality_label.setText("信号质量：合格" if quality_pass else "信号质量：不合格")

        # 這裡把原先的 autoencoder reconstruction error 改成「ECG anomaly score」
        #檢查ECG_RealPeaks的間距(以及跟邊界的距離)，大於3秒(375個採樣點)，則判定為異常，小於0.5秒(約63個採樣點)，則判定為異常
        max_distance_threshold = 375
        min_distance_threshold = 63
        extend_boundary = self.data['ECG_RPeaks'][segment] 
        extend_boundary = np.array([0, *extend_boundary, 1249])
        diff = np.diff(extend_boundary)
        print(f"diff: {diff}, length: {len(diff)}")
        diff2 = diff[1:-1]

        info_text = ""
        # 計算 ECG anomaly score (DTW)
        ecg_anomaly_score = self.compute_ecg_dtw_score(segment)
        if ecg_anomaly_score is not None:
            threshold = 3.0   # 這個閾值可自行調整
            status = "正常" if ecg_anomaly_score < threshold and len(self.data['ECG_RPeaks'][segment]) > 4 and np.all(diff < max_distance_threshold) and np.all(diff2 > min_distance_threshold) else "異常"
            info_text += f"\nECG 異常分數(平均DTW): {ecg_anomaly_score:.2f}\n"
            info_text += f"判定: {status}\n"
            self.anomaly_score_label.setText(f"Anomaly Score: {ecg_anomaly_score:.2f}")
        else:
            self.anomaly_score_label.setText("Anomaly Score: N/A")

        seg_sbp = self.data.get('SegSBP', [])[segment] if 'SegSBP' in self.data else "N/A"
        seg_dbp = self.data.get('SegDBP', [])[segment] if 'SegDBP' in self.data else "N/A"
        info_text += f"SegSBP: {seg_sbp}\nSegDBP: {seg_dbp}\n"

        ppg_speaks = self.data.get('PPG_SPeaks', [])[segment]
        ppg_turns = self.data.get('PPG_Turns', [])[segment]
        ecg_realpeaks = self.data.get('ECG_RPeaks', [])[segment]
        # 可以做各種時間差計算
        info_text += "\n... (各種時間差、血壓資訊略)\n"



        self.info_label.setText(info_text)

        # 將上面訊息一併顯示在 analysis_text
        analysis_text = f"Segment {segment} 分析結果\n"
        analysis_text += "=" * 30 + "\n\n"
        analysis_text += info_text
        self.analysis_text.setText(analysis_text)


    def calculate_second_derivative(self, segment):
        total_segments = len(self.data['ABP_Raw'])
        
        # 获取当前段、前一段和后一段的数据
        current_segment = self.data['ABP_Raw'][segment]
        prev_segment = self.data['ABP_Raw'][segment - 1] if segment > 0 else current_segment
        next_segment = self.data['ABP_Raw'][segment + 1] if segment < total_segments - 1 else current_segment

        # 合并数据，加前后125个点（1秒）
        extended_data = np.concatenate([prev_segment[-125:], current_segment, next_segment[:125]])
        
        # 应用高斯滤波
        smoothed_data = gaussian_filter1d(extended_data, sigma=3)
        
        # 计算二次微分
        second_derivative = np.gradient(np.gradient(smoothed_data))
        
        # 返回中间1250个点（10秒）的结果
        return second_derivative[125:-125]

    def calculate_double_integrated(self, second_derivative):
        cutoff = 0.5  # 截止頻率
        return lowpass_filter(second_derivative, cutoff, 125)

    def plot_reconstruction_error(self, data):
        self.reconstruction_error_plot.clear()

        x = np.arange(len(self.reconstructed_error))
        self.reconstruction_error_plot.plot(x, self.reconstructed_error, pen=pg.mkPen(color=(255, 0, 0), width=2))
        self.reconstruction_error_plot.setYRange(0, np.max(self.reconstructed_error) * 1.1)

    def find_best_alignment(self, seq1, seq2, max_shift=10):
        """
        使用滑動窗口相關性找到兩個interval序列的最佳對齊位置，並返回對齊後的等長序列
        
        參數:
        seq1, seq2: 兩個要對齊的interval序列
        max_shift: 最大允許的偏移量
        
        返回:
        aligned_seq1: 對齊後的序列1
        aligned_seq2: 對齊後的序列2
        best_offset: 最佳對齊的偏移量
        best_correlation: 最佳對齊時的相關係數
        """
        best_correlation = -float('inf')
        best_offset = 0
        best_s1 = None
        best_s2 = None
        
        # 計算兩個序列的長度
        len1, len2 = len(seq1), len(seq2)
        
        # 在允許的偏移範圍內尋找最佳對齊
        for offset in range(-max_shift, max_shift + 1):
            if offset >= 0:
                # seq1向右偏移
                s1 = seq1[offset:]
                s2 = seq2[:len(s1)]
            else:
                # seq2向右偏移
                s2 = seq2[-offset:]
                s1 = seq1[:len(s2)]
                
            if len(s1) < 3 or len(s2) < 3:  # 確保有足夠的數據進行相關性計算
                continue
                
            # 確保兩個序列等長
            min_len = min(len(s1), len(s2))
            s1 = s1[:min_len]
            s2 = s2[:min_len]
                
            # 計算相關係數
            correlation = np.corrcoef(s1, s2)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_offset = offset
                best_s1 = s1
                best_s2 = s2
                
        return best_s1, best_s2, best_offset, best_correlation

    def analyze_ppg_abp_peaks_alignment(self, data):
        """
        分析PPG和ABP peaks間隔序列的差異分布
        """
        ppg_intervals = []
        abp_intervals = []
        
        # 計算每個片段中的間隔
        for segment in range(len(data['PPG_SPeaks'])):
            ppg_peaks = data['PPG_SPeaks'][segment]
            abp_peaks = data['ABP_SPeaks'][segment]
            
            # 計算間隔
            ppg_segment_intervals = np.diff(ppg_peaks)
            abp_segment_intervals = np.diff(abp_peaks)
            
            ppg_intervals.extend(ppg_segment_intervals)
            abp_intervals.extend(abp_segment_intervals)
        
        ppg_intervals = np.array(ppg_intervals)
        abp_intervals = np.array(abp_intervals)
        print(f"ppg_intervals: {ppg_intervals}, length: {len(ppg_intervals)}")
        print(f"abp_intervals: {abp_intervals}, length: {len(abp_intervals)}")
        # 找到最佳對齊位置和對齊後的序列
        aligned_ppg, aligned_abp, best_offset, correlation = self.find_best_alignment(
            ppg_intervals, abp_intervals)
        print(f"aligned_ppg: {aligned_ppg}, length: {len(aligned_ppg)}")
        print(f"aligned_abp: {aligned_abp}, length: {len(aligned_abp)}")
        # 計算對齊後的差異
        interval_differences = np.abs(aligned_ppg - aligned_abp)
        print(f"interval_differences: {interval_differences}, length: {len(interval_differences)}")
        # 計算統計信息
        mean_diff = np.mean(interval_differences)
        std_diff = np.std(interval_differences)
        
        # 將結果添加到meta信息
        self.meta_info.append(f"\nPeaks間隔分析結果:")
        self.meta_info.append(f"最佳對齊偏移量: {best_offset} 個間隔")
        self.meta_info.append(f"對齊相關係數: {correlation:.4f}")
        self.meta_info.append(f"對齊後序列長度: {len(aligned_ppg)}")
        self.meta_info.append(f"平均間隔差異: {mean_diff:.2f} 採樣點")
        self.meta_info.append(f"間隔差異標準差: {std_diff:.2f} 採樣點")
        self.meta_info.append(f"最大間隔差異: {np.max(interval_differences):.2f} 採樣點")
        self.meta_info.append(f"最小間隔差異: {np.min(interval_differences):.2f} 採樣點")
        
        return interval_differences, best_offset, correlation

    def plot_alignment_results(self, interval_differences, best_offset, correlation):
        """
        在新窗口中繪製間隔差異的分析結果
        """
        fig = plt.figure(figsize=(15, 5))
        
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建三個子圖
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        
        # 繪製直方圖
        ax1.hist(interval_differences, bins=50, edgecolor='black')
        ax1.set_title(f'間隔差異分布\n(offset={best_offset}, corr={correlation:.3f})')
        ax1.set_xlabel('間隔差異（採樣點）')
        ax1.set_ylabel('頻率')
        ax1.grid(True)
        
        # 繪製箱型圖
        ax2.boxplot(interval_differences)
        ax2.set_title('間隔差異箱型圖')
        ax2.set_ylabel('間隔差異（採樣點）')
        ax2.grid(True)
        
        # 繪製差異的時間序列
        ax3.plot(interval_differences, 'b-', alpha=0.6)
        ax3.set_title('間隔差異時間序列')
        ax3.set_xlabel('間隔序號')
        ax3.set_ylabel('差異（採樣點）')
        ax3.grid(True)
        
        # 添加統計信息
        stats_text = f'平均差異: {np.mean(interval_differences):.2f}\n'
        stats_text += f'標準差: {np.std(interval_differences):.2f}\n'
        stats_text += f'中位數: {np.median(interval_differences):.2f}'
        ax2.text(1.4, np.median(interval_differences), stats_text, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        timer = QTimer()
        timer.singleShot(100, lambda: plt.show())

    def analyze_peaks_alignment(self):
        if 'PPG_SPeaks' in self.data and 'ABP_SPeaks' in self.data:
            interval_differences, best_offset, correlation = self.analyze_ppg_abp_peaks_alignment(self.data)
            if interval_differences is not None:
                self.plot_alignment_results(interval_differences, best_offset, correlation)
        else:
            self.meta_info.append("\n錯誤：找不到PPG或ABP peaks數據")

    def find_real_peaks(self, signal, original_peaks, window_size=5):
        """
        從original_peaks位置開始，根據信號變化方向尋找最近的局部極大值
        
        參數:
        signal: ECG_F信號
        original_peaks: 原始標註的peaks位置
        window_size: 當在極小值時，搜尋窗口大小（前後各window_size個點）
        
        返回:
        real_peaks: 修正後的peaks位置
        """
        real_peaks = []
        signal_len = len(signal)
        
        for peak in original_peaks:
            if peak >= signal_len:
                continue
            
            # 檢查是否已經在局部極大值
            if peak > 0 and peak < signal_len-1:
                if signal[peak] >= signal[peak-1] and signal[peak] >= signal[peak+1]:
                    real_peaks.append(peak)  # 已經在極大值，不需要移動
                    continue
                
            # 檢查是否在極小值附近（需要在窗口內尋找最大值）
            if peak > 0 and peak < signal_len-1:
                if signal[peak] <= signal[peak-1] and signal[peak] <= signal[peak+1]:
                    # 在極小值，使用窗口搜尋
                    start = max(0, peak - window_size)
                    end = min(signal_len, peak + window_size + 1)
                    window = signal[start:end]
                    local_max_idx = start + np.argmax(window)
                    real_peaks.append(local_max_idx)
                    continue
            
            # 其他情況：根據斜率方向尋找最近的極大值
            left_idx = peak
            right_idx = peak
            
            # 向左搜尋
            while left_idx > 0:
                if signal[left_idx-1] > signal[left_idx]:
                    left_idx -= 1
                else:
                    break
                
            # 向右搜尋
            while right_idx < signal_len-1:
                if signal[right_idx+1] > signal[right_idx]:
                    right_idx += 1
                else:
                    break
                
            # 比較左右兩側找到的點，選擇信號值較大的
            if signal[left_idx] > signal[right_idx]:
                real_peaks.append(left_idx)
            else:
                real_peaks.append(right_idx)
        
        return np.array(real_peaks)

    def calculate_time_differences(self, ppg_points, ecg_points, fs):
        time_diffs = []
        for ppg_idx in ppg_points:
            # 找到小于 ppg_idx 的最大 ecg_idx
            ecg_prior = ecg_points[ecg_points <= ppg_idx]
            if len(ecg_prior) == 0:
                continue  # 没有找到之前的 ECG_RealPeaks
            ecg_idx = ecg_prior[-1]
            time_diff = (ppg_idx - ecg_idx) / fs
            time_diffs.append(time_diff)
        return time_diffs

    def calculate_reconstruction_error(self, segment_idx):
        """計算指定片段的重構誤差"""
        try:
            # 獲取 ECG 數據
            ecg_data = self.data['ECG_F'][segment_idx]
            
            # 直接轉換為 tensor
            ecg_tensor = torch.FloatTensor(ecg_data).to(self.device)
            ecg_tensor = ecg_tensor.view(1, 1, -1)  # 添加批次和通道維度
            
            # 計算重構誤差
            with torch.no_grad():
                reconstructed = self.ecg_model(ecg_tensor)
                error = F.mse_loss(reconstructed, ecg_tensor).item()
                
            return error
            
        except Exception as e:
            print(f"計算重構誤差失敗: {str(e)}")
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PulseDBViewer()
    window.show()
    sys.exit(app.exec_())