import sys
import os
import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QCheckBox, QComboBox, QSlider, QLabel, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
# from model_abp_1250points import ConvAutoencoder, ConvAutoencoder2, predict_reconstructed_1250abp
# from model_pulse_representation import predict_reconstructed_abp
# from abp_anomaly_detection_algorithms import detect_anomaly_gui, calculate_dtw_scores
from load_normap_ABP_data import load_training_set
import pickle
# import torch
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from prepare_training_data_vitaldb import check_signal_quality
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

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            self.scaleBy((1.1**(-ev.delta() * 0.001), 1), center=(0.5, 0.5))
        else:
            self.scaleBy((1, 1.1**(-ev.delta() * 0.001)), center=(0.5, 0.5))

class PulseDBViewer(QMainWindow):
    def __init__(self):
        super().__init__()
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
            self.checkboxes[signal].setChecked(False) if 'ABP' in signal else self.checkboxes[signal].setChecked(True)
            self.checkboxes[signal].stateChanged.connect(self.update_plot)
            left_layout.addWidget(self.checkboxes[signal])

        # 元信息顯示區域
        self.meta_info = QTextEdit()
        self.meta_info.setReadOnly(True)
        left_layout.addWidget(self.meta_info)

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
        with h5py.File(file_path, 'r') as f:
            matdata = f['Subj_Wins']
            self.data = {}
            for key in matdata.keys():
                if key in ['ABP_Raw', 'ABP_F', 'ECG_Raw', 'ECG_F', 'PPG_Raw', 'PPG_F']:
                    self.data[key] = [f[ref][:].flatten() for ref in matdata[key][0]]
                elif key in ['ABP_SPeaks', 'ABP_Turns', 'ECG_RPeaks', 'PPG_SPeaks', 'PPG_Turns']:
                    self.data[key] = [f[ref][:].flatten().astype(int) for ref in matdata[key][0]]
                elif key in ['SegmentID', 'CaseID', 'SubjectID']:
                    # 印出 SegmentID, CaseID, SubjectID 的實際值
                    self.data[key] = []
                    for ref in matdata[key][0]:
                        value = f[ref][()]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')  # 解碼為字符串
                        elif isinstance(value, np.ndarray):
                            value = value.tolist()  # 轉換為列表
                        self.data[key].append(value)
                    print(f"{key}: {self.data[key]}")
                else:
                    # 其他欄位印出形狀
                    self.data[key] = [f[ref][:] for ref in matdata[key][0]]
                    print(f"{key} 的形狀: {matdata[key].shape}")

        self.segment_slider.setMaximum(len(self.data['ABP_Raw']) - 1)
        self.update_meta_info()
        self.update_plot()
        # 在加载数据后初始化重构误差图
        if 'ABP_Raw' in self.data and len(self.data['ABP_Raw']) > 0:
            self.plot_reconstruction_error(self.data['ABP_Raw'][0])
        else:
            print("警告: 文件中未找到 ABP_Raw 数据。")


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
        for key in ['Age', 'BMI', 'CaseID', 'Gender', 'Height', 'IncludeFlag', 'SegDBP', 'SegSBP', 'SegmentID', 'SubjectID', 'Weight', 'WinID', 'WinSeqID']:
            if key in self.data:
                meta_text += f"{key}: {self.data[key][0][0]}\n"
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


    def update_plot(self):
        self.plot_widget.clear()
        self.normalized_plot_widget.clear()
        segment = self.segment_slider.value()
        self.segment_label.setText(f"Segment: {segment}")

        if self.db_combo.currentText() == "training_set.npz":
            if self.training_abp_data is not None:
                x = np.arange(1250)  # 10 seconds at 125 Hz
                y = self.training_abp_data[segment]
                self.plot_widget.plot(x, y, pen=pg.mkPen(color=(255, 0, 0), width=2))

                # 顯示預先計算好的異常分數
                self.anomaly_score_label.setText(f"Anomaly Score: {self.anomaly_scores[segment]:.4f}")
                # self.svm_score_label.setText(f"SVM Score: {self.svm_scores[segment]:.4f}")
                # self.if_score_label.setText(f"IF Score: {self.if_scores[segment]:.4f}")
                turns = self.training_abp_turns[segment]
                # dtws = calculate_dtw_scores(y, turns)
                # self.update_dtw_plot(dtws, turns)

        else:
            if not self.data:
                return

            x = np.arange(1250)  # 10 seconds at 125 Hz
            colors = {
                'ABP': (255, 0, 0),          # 红色
                'ABP_SPeaks': (0, 255, 0),   # 绿色
                'ABP_Turns': (0, 0, 255),    # 蓝色
                'ECG': (0, 255, 0),      # 綠色
                'PPG': (0, 0, 255),      # 藍色
                'ECG_peaks': (255, 0, 255),  # 洋紅色
                'ECG_real_peaks': (255, 255, 0),  # 黃色
                'PPG_peaks': (255, 165, 0),   # 橙色
                'PPG_turns': (0, 255, 255)    # 青色
            }

            # 处理 ABP_Raw 信号及其特征点
            if self.checkboxes['ABP_Raw'].isChecked():
                y_raw = self.data['ABP_Raw'][segment]
                self.plot_widget.plot(x, y_raw, pen=pg.mkPen(color=colors['ABP'], width=2, name='ABP_Raw'))
                
                # 在 ABP_Raw 上显示 ABP_SPeaks
                if self.checkboxes['ABP_SPeaks'].isChecked() and 'ABP_SPeaks' in self.data:
                    peaks = self.data['ABP_SPeaks'][segment]
                    peaks = peaks[peaks < len(y_raw)]
                    y_peaks = y_raw[peaks]
                    self.plot_widget.plot(peaks, y_peaks, pen=None, symbol='o', 
                                        symbolPen=None, symbolSize=8, 
                                        symbolBrush=colors['ABP_SPeaks'])

                # 在 ABP_Raw 上显示 ABP_Turns
                if self.checkboxes['ABP_Turns'].isChecked() and 'ABP_Turns' in self.data:
                    turns = self.data['ABP_Turns'][segment]
                    turns = turns[turns < len(y_raw)]
                    y_turns = y_raw[turns]
                    self.plot_widget.plot(turns, y_turns, pen=None, symbol='s', 
                                        symbolPen=None, symbolSize=8, 
                                        symbolBrush=colors['ABP_Turns'])

            # 处理 ABP_F 信号及其特征点
            if self.checkboxes['ABP_F'].isChecked():
                y_norm = self.data['ABP_F'][segment]
                self.normalized_plot_widget.plot(x, y_norm, 
                                            pen=pg.mkPen(color=colors['ABP'], width=2, 
                                            name='ABP_F'))
                
                # 在 ABP_F 上显示 ABP_SPeaks
                if self.checkboxes['ABP_SPeaks'].isChecked() and 'ABP_SPeaks' in self.data:
                    peaks = self.data['ABP_SPeaks'][segment]
                    peaks = peaks[peaks < len(y_norm)]
                    y_peaks = y_norm[peaks]
                    self.normalized_plot_widget.plot(peaks, y_peaks, pen=None, symbol='o', 
                                                symbolPen=None, symbolSize=8, 
                                                symbolBrush=colors['ABP_SPeaks'])

                # 在 ABP_F 上显示 ABP_Turns
                if self.checkboxes['ABP_Turns'].isChecked() and 'ABP_Turns' in self.data:
                    turns = self.data['ABP_Turns'][segment]
                    turns = turns[turns < len(y_norm)]
                    y_turns = y_norm[turns]
                    self.normalized_plot_widget.plot(turns, y_turns, pen=None, symbol='s', 
                                                symbolPen=None, symbolSize=8, 
                                                symbolBrush=colors['ABP_Turns'])
            # 處理ECG信號
            if self.checkboxes['ECG_Raw'].isChecked():
                y = self.data['ECG_Raw'][segment]
                self.plot_widget.plot(x, y, pen=pg.mkPen(color=colors['ECG'], width=2, name='ECG_Raw'))
                
                if self.checkboxes['ECG_RPeaks'].isChecked():
                    peaks = self.data['ECG_RPeaks'][segment]
                    peaks = peaks[peaks < 1250]
                    print(f'ECG peaks: {peaks}, length: {len(peaks)}')
                    y_peaks = y[peaks]
                    self.plot_widget.plot(peaks, y_peaks, pen=None, symbol='o', 
                                        symbolPen=None, symbolSize=2, symbolBrush=colors['ECG'])

            # 在normalized_plot_widget中顯示正規化的信號和peaks
            for signal_type in ['ABP', 'ECG', 'PPG']:
                if self.checkboxes[f'{signal_type}_F'].isChecked():
                    y_norm = self.data[f'{signal_type}_F'][segment]
                    self.normalized_plot_widget.plot(x, y_norm, 
                                                   pen=pg.mkPen(color=colors[signal_type], width=2, 
                                                   name=f'{signal_type}_F'))
                    
                    # ECG peaks 的處理
                    if signal_type == 'ECG' and self.checkboxes['ECG_RPeaks'].isChecked():
                        # 原始peaks（紅色）
                        original_peaks = self.data['ECG_RPeaks'][segment]
                        original_peaks = original_peaks[original_peaks < 1250]
                        y_peaks = y_norm[original_peaks]
                        
                        # 顯示原始peaks
                        scatter_original = pg.ScatterPlotItem(
                            original_peaks, y_peaks,
                            symbol='o',
                            size=10,
                            pen=pg.mkPen(colors['ECG_peaks'], width=2),
                            brush=pg.mkBrush(colors['ECG_peaks']),
                            name='Original_ECG_Peaks'
                        )
                        self.normalized_plot_widget.addItem(scatter_original)
                        
                        # 尋找真實peaks（黃色）
                        real_peaks = self.find_real_peaks(y_norm, original_peaks)
                        y_real_peaks = y_norm[real_peaks]
                        
                        # 顯示真實peaks
                        scatter_real = pg.ScatterPlotItem(
                            real_peaks, y_real_peaks,
                            symbol='x',  # 使用不同的符號以區分
                            size=15,
                            pen=pg.mkPen(colors['ECG_real_peaks'], width=3),
                            name='Real_ECG_Peaks'
                        )
                        self.normalized_plot_widget.addItem(scatter_real)
                        
                        # 添加垂直線標示兩種peaks的位置
                        for peak, real_peak in zip(original_peaks, real_peaks):
                            # 原始peak位置（紅色虛線）
                            line_original = pg.InfiniteLine(
                                pos=peak, 
                                angle=90, 
                                pen=pg.mkPen(color=colors['ECG_peaks'], width=1, style=Qt.DashLine)
                            )
                            # 真實peak位置（黃色虛線）
                            line_real = pg.InfiniteLine(
                                pos=real_peak, 
                                angle=90, 
                                pen=pg.mkPen(color=colors['ECG_real_peaks'], width=1, style=Qt.DashLine)
                            )
                            self.normalized_plot_widget.addItem(line_original)
                            self.normalized_plot_widget.addItem(line_real)
                        
                        # 顯示peaks的差異統計
                        diff = np.abs(real_peaks - original_peaks)
                        avg_diff = np.mean(diff)
                        max_diff = np.max(diff)
                        self.meta_info.append(f"\nECG Peaks 差異統計:")
                        self.meta_info.append(f"平均偏移: {avg_diff:.2f} 點")
                        self.meta_info.append(f"最大偏移: {max_diff:.2f} 點")

                    # PPG peaks 和 turns 的處理
                    if signal_type == 'PPG':
                        # 處理 PPG peaks
                        if self.checkboxes['PPG_SPeaks'].isChecked():
                            peaks = self.data['PPG_SPeaks'][segment]
                            peaks = peaks[peaks < 1250]
                            y_peaks = y_norm[peaks]
                            
                            scatter_peaks = pg.ScatterPlotItem(
                                peaks, y_peaks,
                                symbol='o',
                                size=10,
                                pen=pg.mkPen(colors['PPG_peaks'], width=2),
                                brush=pg.mkBrush(colors['PPG_peaks']),
                                name='PPG_Peaks'
                            )
                            self.normalized_plot_widget.addItem(scatter_peaks)
                            
                            # 可選：添加垂直線標示peak位置
                            for peak in peaks:
                                line = pg.InfiniteLine(
                                    pos=peak, 
                                    angle=90, 
                                    pen=pg.mkPen(color=colors['PPG_peaks'], width=1, style=Qt.DashLine)
                                )
                                self.normalized_plot_widget.addItem(line)

                        # 處理 PPG turns
                        if self.checkboxes['PPG_Turns'].isChecked():
                            turns = self.data['PPG_Turns'][segment]
                            turns = turns[turns < 1250]
                            y_turns = y_norm[turns]
                            
                            scatter_turns = pg.ScatterPlotItem(
                                turns, y_turns,
                                symbol='s',  # 使用方形標記區分turns
                                size=8,
                                pen=pg.mkPen(colors['PPG_turns'], width=2),
                                brush=pg.mkBrush(colors['PPG_turns']),
                                name='PPG_Turns'
                            )
                            self.normalized_plot_widget.addItem(scatter_turns)
                            
                            # 可選：添加垂直線標示turns位置
                            for turn in turns:
                                line = pg.InfiniteLine(
                                    pos=turn, 
                                    angle=90, 
                                    pen=pg.mkPen(color=colors['PPG_turns'], width=1, style=Qt.DashLine)
                                )
                                self.normalized_plot_widget.addItem(line)

            if 'ABP_Raw' in self.data and 'ABP_Turns' in self.data and 'ABP_SPeaks' in self.data:
                abp_data = self.data['ABP_Raw'][segment]
                if self.checkbox_smoothed.isChecked():
                    smoothed_abp = gaussian_filter1d(abp_data, sigma=3)
                    self.plot_widget.plot(x, smoothed_abp, 
                                        pen=pg.mkPen(color=(100, 155, 155), width=2, name='Smoothed ABP'))
                    
                    if self.checkbox_second_derivative.isChecked():
                        second_derivative = self.calculate_second_derivative(segment)
                        self.normalized_plot_widget.plot(x, second_derivative, 
                                                       pen=pg.mkPen(color=(255, 0, 255), width=2, 
                                                       name='ABP Second Derivative'))

                    if self.checkbox_double_integrated.isChecked():
                        double_integrated = self.calculate_double_integrated(second_derivative)
                        self.normalized_plot_widget.plot(x, double_integrated, 
                                                       pen=pg.mkPen(color=(0, 0, 0), width=2, 
                                                       name='ABP Double Integrated'))

            # 更新Peak資訊
            self.update_peak_info()
            
            # 在更新绘图后，进行信号质量检查并更新界面显示
            peaks_dict = {
                'ECG_RealPeaks': self.data.get('ECG_RPeaks', [])[segment],
                'PPG_SPeaks': self.data.get('PPG_SPeaks', [])[segment],
                'PPG_Turns': self.data.get('PPG_Turns', [])[segment],
            }

            # 检查特征点是否存在
            if any(len(peaks) == 0 for peaks in peaks_dict.values()):
                self.quality_label.setText("信号质量：不合格（缺少特征点）")
            else:
                quality_pass = check_signal_quality(peaks_dict)
                if quality_pass:
                    self.quality_label.setText("信号质量：合格")
                else:
                    self.quality_label.setText("信号质量：不合格")

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
        # double_integrated = scipy.integrate.cumtrapz(scipy.integrate.cumtrapz(second_derivative, initial=0), initial=0)
        # return double_integrated

    # def calculate_anomaly_score(self):
    #     if self.db_combo.currentText() != "training_set.npz" and 'ABP_Raw' in self.data and 'ABP_Turns' in self.data and 'ABP_SPeaks' in self.data:
    #         segment = self.segment_slider.value()
    #         data = self.data['ABP_Raw'][segment]
    #         turns = self.data['ABP_Turns'][segment]
    #         speaks = self.data['ABP_SPeaks'][segment]
    #         anomaly_score = detect_anomaly_gui(data, turns, speaks)
    #         self.anomaly_score_label.setText(f"Anomaly Score: {anomaly_score:.4f}")
    #     else:
    #         self.anomaly_score_label.setText("Anomaly Score: N/A")

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PulseDBViewer()
    window.show()
    sys.exit(app.exec_())