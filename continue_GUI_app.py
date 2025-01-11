import sys
import os
import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QListWidget, QCheckBox, QLabel, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt

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

class ContinuousSignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuous Signal Viewer")
        self.resize(1400, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # 左側面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 數據庫選擇
        self.db_combo = QComboBox()
        self.db_combo.addItems(["PulseDB_Vital", "PulseDB_MIMIC", "processed_data"])
        self.db_combo.currentIndexChanged.connect(self.load_files)
        left_layout.addWidget(self.db_combo)

        # 文件列表
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_file)
        left_layout.addWidget(self.file_list)

        # 信號選擇複選框
        self.checkboxes = {}
        for signal in ['ABP_Raw', 'ABP_F', 'ABP_SPeaks', 'ECG_Raw', 'ECG_F', 'ECG_RPeaks', 
                      'PPG_Raw', 'PPG_F', 'PPG_SPeaks']:
            self.checkboxes[signal] = QCheckBox(signal)
            self.checkboxes[signal].setChecked(True) #if 'ABP' in signal else self.checkboxes[signal].setChecked(False)
            self.checkboxes[signal].stateChanged.connect(self.update_plot)
            left_layout.addWidget(self.checkboxes[signal])

        # 元信息顯示
        self.meta_info = QTextEdit()
        self.meta_info.setReadOnly(True)
        left_layout.addWidget(self.meta_info)

        # 右側面板 - 繪圖區域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 創建兩個PlotWidget用於顯示原始和正規化信號
        self.plot_widget = pg.PlotWidget(viewBox=CustomViewBox())
        self.normalized_plot_widget = pg.PlotWidget(viewBox=CustomViewBox())

        for widget in [self.plot_widget, self.normalized_plot_widget]:
            widget.setBackground('w')
            widget.setLabel('left', 'Amplitude')
            widget.setLabel('bottom', 'Time (seconds)')
            widget.setMenuEnabled(False)
            widget.showGrid(x=True, y=True)

        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.normalized_plot_widget)

        # 添加左側和右側面板
        self.main_layout.addWidget(left_panel, 1)
        self.main_layout.addWidget(right_panel, 4)

        self.data = {}
        self.load_files()

    def load_files(self):
        selected_db = self.db_combo.currentText()
        folder_path = os.path.join("D:\\PulseDB", selected_db)
        self.file_list.clear()
        
        # 根據選擇的數據庫類型列出對應的文件
        if selected_db == "processed_data":
            for file in os.listdir(folder_path):
                if file.endswith(".h5"):
                    self.file_list.addItem(file)
        else:
            for file in os.listdir(folder_path):
                if file.endswith(".mat"):
                    self.file_list.addItem(file)

    def load_selected_file(self, item):
        db_folder = self.db_combo.currentText()
        file_path = os.path.join("D:\\PulseDB", db_folder, item.text())
        
        if file_path.endswith('.h5'):
            self.load_h5_file(file_path)
        else:
            self.load_mat_file(file_path)

    def load_mat_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            matdata = f['Subj_Wins']
            self.data = {}
            
            # 讀取所有片段並連接
            for key in matdata.keys():
                if key in ['ABP_Raw', 'ABP_F', 'ECG_Raw', 'ECG_F', 'PPG_Raw', 'PPG_F']:
                    # 連接所有片段
                    segments = [f[ref][:].flatten() for ref in matdata[key][0]]
                    self.data[key] = np.concatenate(segments)
                    print(f"{key} shape: {self.data[key].shape}")  # 打印形狀
                elif key in ['ABP_SPeaks', 'ECG_RPeaks', 'PPG_SPeaks']:
                    # 對於peaks，需要調整索引以反映連續信號
                    peaks = []
                    segment_length = 1250  # 10秒 * 125Hz
                    for i, ref in enumerate(matdata[key][0]):
                        segment_peaks = f[ref][:].flatten().astype(int)
                        # 加上對應片段的偏移
                        peaks.extend(segment_peaks + i * segment_length)
                    self.data[key] = np.array(peaks)
                    print(f"{key} shape: {self.data[key].shape}")  # 打印形狀
                else:
                    # 添加打印指定元數據內容
                    if key in ['CaseID', 'SegmentID', 'SubjectID']:
                        print(f"{key} contents:\n{matdata[key]}")  # 打印內容
                    # 其他元數據合併為一個陣列
                    metadata = [f[ref][:].flatten() for ref in matdata[key][0]]
                    try:
                        self.data[key] = np.array(metadata)
                        print(f"{key} shape: {self.data[key].shape}")  # 打印整體形狀
                    except Exception as e:
                        print(f"Warning: Could not load {key}: {e}")
                    


            self.update_meta_info()
            self.update_plot()

    def load_h5_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.data = {}
            
            # 讀取所有數據集
            for key in f.keys():
                if key == 'annotations':
                    self.data[key] = f[key][:]
                    print(f"{key} shape: {self.data[key].shape}")
                    continue
                    
                if key in ['ABP_Raw', 'ABP_F', 'ECG_Raw', 'ECG_F', 'PPG_Raw', 'PPG_F']:
                    segments = f[key][:]
                    if len(segments.shape) > 1:  # 如果是二維數組
                        self.data[key] = segments.flatten() if segments.shape[1] == 1 else segments.reshape(-1)
                    else:
                        self.data[key] = segments
                    print(f"{key} shape: {self.data[key].shape}")
                    
                elif key in ['ABP_SPeaks', 'ECG_RPeaks', 'PPG_SPeaks']:
                    peaks = []
                    segment_length = 1250
                    segments = f[key][:]
                    for i, segment_peaks in enumerate(segments):
                        if isinstance(segment_peaks, np.ndarray):
                            peaks.extend(segment_peaks + i * segment_length)
                        else:
                            peaks.append(segment_peaks + i * segment_length)
                    self.data[key] = np.array(peaks)
                    print(f"{key} shape: {self.data[key].shape}")
                    
                else:
                    try:
                        metadata = [f[key][ref][:].flatten() for ref in matdata[key][0]]
                        self.data[key] = np.array(metadata)
                        print(f"{key} shape: {self.data[key].shape}")
                        
                        # 添加打印指定元數據內容
                        if key in ['CaseID', 'SegmentID', 'SubjectID']:
                            print(f"{key} contents:\n{self.data[key]}")  # 打印內容
                            
                    except Exception as e:
                        print(f"Warning: Could not load {key}: {e}")
        
        # 打印數據形狀以進行調試
        for key in self.data:
            if isinstance(self.data[key], np.ndarray):
                print(f"{key} shape: {self.data[key].shape}")
            elif isinstance(self.data[key], list):
                for idx, item in enumerate(self.data[key]):
                    print(f"{key}[{idx}] shape: {item.shape}")
        
        self.update_meta_info()
        self.update_plot()

    def update_meta_info(self):
        meta_text = "檔案信息:\n"
        
        # 顯示基本信息
        for key in ['Age', 'BMI', 'CaseID', 'Gender', 'Height', 'SegDBP', 'SegSBP', 
                   'SubjectID', 'Weight']:
            if key in self.data:
                if isinstance(self.data[key], np.ndarray):
                    value = self.data[key].flatten()[0] if self.data[key].shape[1] == 1 else self.data[key]
                else:
                    value = self.data[key][0][0]
                meta_text += f"{key}: {value}\n"
        
        # 添加信號長度信息
        if 'ABP_Raw' in self.data:
            duration_seconds = len(self.data['ABP_Raw']) / 125  # 採樣率125Hz
            meta_text += f"\n總時長: {duration_seconds:.2f} 秒"
            meta_text += f"\n總採樣點: {len(self.data['ABP_Raw'])}"
        
        # 顯示 CaseID、SegmentID 和 SubjectID 的完整內容
        for key in ['CaseID', 'SegmentID', 'SubjectID']:
            if key in self.data:
                meta_text += f"\n\n{key} contents:\n{self.data[key]}"
        
        self.meta_info.setText(meta_text)

    def update_plot(self):
        self.plot_widget.clear()
        self.normalized_plot_widget.clear()

        if not self.data:
            return

        # 創建時間軸（以秒為單位）
        sampling_rate = 125  # Hz
        time = np.arange(len(self.data['ABP_Raw'])) / sampling_rate

        colors = {'ABP': (255, 0, 0), 'ECG': (0, 255, 100), 'PPG': (0, 0, 255)}

        # 在上方視窗顯示 ABP 相關信號
        if self.checkboxes['ABP_Raw'].isChecked():
            y = self.data['ABP_Raw']
            self.plot_widget.plot(time, y, pen=pg.mkPen(color=colors['ABP'], width=1))

            # 在 ABP_Raw 上標註peaks
            if self.checkboxes['ABP_SPeaks'].isChecked() and 'ABP_SPeaks' in self.data:
                peaks = self.data['ABP_SPeaks']
                peaks_time = peaks / sampling_rate
                peaks_values = y[peaks]
                self.plot_widget.plot(peaks_time, peaks_values, pen=None, 
                                    symbol='o', symbolSize=4, 
                                    symbolBrush=colors['ABP'])

        if self.checkboxes['ABP_F'].isChecked():
            y_norm = self.data['ABP_F']
            self.plot_widget.plot(time, y_norm, 
                                pen=pg.mkPen(color=colors['ABP'], width=1, style=Qt.DashLine))

        # 在下方視窗顯示 ECG 和 PPG 相關信號
        for signal_type in ['ECG', 'PPG']:
            if self.checkboxes[f'{signal_type}_Raw'].isChecked():
                y = self.data[f'{signal_type}_Raw']
                self.normalized_plot_widget.plot(time, y, 
                                               pen=pg.mkPen(color=colors[signal_type], width=1))

                # 在原始信號上標註peaks
                peaks_key = f'{signal_type}_RPeaks' if signal_type == 'ECG' else f'{signal_type}_SPeaks'
                if self.checkboxes[peaks_key].isChecked() and peaks_key in self.data:
                    peaks = self.data[peaks_key]
                    peaks_time = peaks / sampling_rate
                    peaks_values = y[peaks]
                    self.normalized_plot_widget.plot(peaks_time, peaks_values, 
                                                   pen=None, symbol='o', symbolSize=5, 
                                                   symbolBrush=colors[signal_type])

            if self.checkboxes[f'{signal_type}_F'].isChecked():
                y_norm = self.data[f'{signal_type}_F']
                self.normalized_plot_widget.plot(time, y_norm, 
                                               pen=pg.mkPen(color=colors[signal_type], 
                                                          width=3, style=Qt.DashLine))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ContinuousSignalViewer()
    window.show()
    sys.exit(app.exec_())