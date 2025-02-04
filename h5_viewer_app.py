import sys
import h5py
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

class H5DataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('H5 Training Data Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化變數
        self.current_file = None
        self.data_cache = {}
        self.max_cache_size = 10
        
        # 設置主要佈局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左側控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        
        # 文件選擇
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_h5_file)
        left_layout.addWidget(QLabel("H5 檔案列表:"))
        left_layout.addWidget(self.file_list)
        
        # Segment 控制
        self.segment_slider = QSlider(Qt.Horizontal)
        self.segment_slider.valueChanged.connect(self.update_display)
        self.segment_label = QLabel("Segment: 0")
        left_layout.addWidget(self.segment_label)
        left_layout.addWidget(self.segment_slider)
        
        # 跳轉控制
        jump_layout = QHBoxLayout()
        self.segment_jump = QSpinBox()
        self.segment_jump.valueChanged.connect(self.jump_to_segment)
        jump_layout.addWidget(QLabel("跳轉到:"))
        jump_layout.addWidget(self.segment_jump)
        left_layout.addLayout(jump_layout)
        
        # 信號顯示控制
        self.signal_group = QGroupBox("信號顯示")
        signal_layout = QVBoxLayout()
        self.checkboxes = {}
        for signal in ['ppg', 'ecg', 'abp']:
            self.checkboxes[signal] = QCheckBox(signal.upper())
            self.checkboxes[signal].setChecked(True)
            self.checkboxes[signal].stateChanged.connect(self.update_display)
            signal_layout.addWidget(self.checkboxes[signal])
        self.signal_group.setLayout(signal_layout)
        left_layout.addWidget(self.signal_group)
        
        # 分析結果顯示
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(200)
        left_layout.addWidget(QLabel("分析結果:"))
        left_layout.addWidget(self.analysis_text)
        
        # 右側圖表面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 創建五個獨立的圖表
        # ECG 圖表
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setBackground('w')
        self.ecg_plot.showGrid(x=True, y=True)
        self.ecg_plot.setTitle("ECG Signal")
        self.ecg_plot.setLabel('left', 'Amplitude (mV)')
        right_layout.addWidget(self.ecg_plot, stretch=1)
        
        # PPG 圖表
        self.ppg_plot = pg.PlotWidget()
        self.ppg_plot.setBackground('w')
        self.ppg_plot.showGrid(x=True, y=True)
        self.ppg_plot.setTitle("PPG Signal")
        self.ppg_plot.setLabel('left', 'Amplitude (a.u.)')
        right_layout.addWidget(self.ppg_plot, stretch=1)
        
        # PPG 一階差分圖表
        self.ppg_first_diff_plot = pg.PlotWidget()
        self.ppg_first_diff_plot.setBackground('w')
        self.ppg_first_diff_plot.showGrid(x=True, y=True)
        self.ppg_first_diff_plot.setTitle("PPG First Derivative")
        self.ppg_first_diff_plot.setLabel('left', 'Amplitude/dt')
        right_layout.addWidget(self.ppg_first_diff_plot, stretch=1)
        
        # PPG 二階差分圖表
        self.ppg_second_diff_plot = pg.PlotWidget()
        self.ppg_second_diff_plot.setBackground('w')
        self.ppg_second_diff_plot.showGrid(x=True, y=True)
        self.ppg_second_diff_plot.setTitle("PPG Second Derivative")
        self.ppg_second_diff_plot.setLabel('left', 'Amplitude/dt²')
        right_layout.addWidget(self.ppg_second_diff_plot, stretch=1)
        
        # ABP 圖表
        self.abp_plot = pg.PlotWidget()
        self.abp_plot.setBackground('w')
        self.abp_plot.showGrid(x=True, y=True)
        self.abp_plot.setTitle("ABP Signal")
        self.abp_plot.setLabel('left', 'Pressure (mmHg)')
        right_layout.addWidget(self.abp_plot, stretch=1)
        
        # 設置 X 軸鏈接
        self.ecg_plot.setXLink(self.ppg_plot)
        self.ppg_plot.setXLink(self.ppg_first_diff_plot)
        self.ppg_first_diff_plot.setXLink(self.ppg_second_diff_plot)
        self.ppg_second_diff_plot.setXLink(self.abp_plot)
        
        # 為所有圖表添加局部放大功能
        for plot in [self.ecg_plot, self.ppg_plot, self.ppg_first_diff_plot, 
                    self.ppg_second_diff_plot, self.abp_plot]:
            plot.setMouseEnabled(x=True, y=True)
            plot.enableAutoRange()
            plot.setAutoVisible(y=True)
            
            # 添加十字準線
            crosshair_pen = pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
            vLine = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
            hLine = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
            plot.addItem(vLine, ignoreBounds=True)
            plot.addItem(hLine, ignoreBounds=True)
            
            def mouseMoved(evt):
                if plot.sceneBoundingRect().contains(evt):
                    mousePoint = plot.plotItem.vb.mapSceneToView(evt)
                    vLine.setPos(mousePoint.x())
                    hLine.setPos(mousePoint.y())
            
            plot.scene().sigMouseMoved.connect(mouseMoved)
        
        # 添加到主佈局
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        # 狀態欄
        self.statusBar().showMessage('就緒')
        
        # 初始化文件列表
        self.update_file_list()
        
    def update_file_list(self):
        """更新文件列表"""
        import glob
        h5_files = glob.glob("personalized_training_data_VitalDB/*.h5")
        self.file_list.clear()
        for file in h5_files:
            self.file_list.addItem(file)
    
    def load_h5_file(self, item):
        """載入 H5 文件"""
        try:
            file_path = item.text()
            self.statusBar().showMessage(f'載入文件: {file_path}')
            
            with h5py.File(file_path, 'r') as f:
                # 獲取數據集大小
                n_samples = f['ppg'].shape[0]
                self.segment_slider.setMaximum(n_samples - 1)
                self.segment_jump.setMaximum(n_samples - 1)
                
                self.current_file = file_path
                self.data_cache.clear()
                
                # 載入第一個片段
                self.load_segment(0)
                self.update_display()
                
            self.statusBar().showMessage(f'已載入: {file_path}, 共 {n_samples} 個片段')
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入文件失敗: {str(e)}")
            self.statusBar().showMessage('載入失敗')
    
    def load_segment(self, idx):
        """載入特定片段的數據"""
        if self.current_file is None or idx in self.data_cache:
            return
            
        try:
            with h5py.File(self.current_file, 'r') as f:
                # 清理緩存
                if len(self.data_cache) >= self.max_cache_size:
                    oldest = min(self.data_cache.keys())
                    del self.data_cache[oldest]
                
                # 載入數據
                segment_data = {
                    'ppg': f['ppg'][idx],
                    'ecg': f['ecg'][idx],
                    'abp': f['abp'][idx],
                    'annotations': f['annotations'][idx],
                    'segsbp': f['segsbp'][idx],
                    'segdbp': f['segdbp'][idx],
                    'personal_info': f['personal_info'][idx],
                    'vascular_properties': f['vascular_properties'][idx]
                }
                
                self.data_cache[idx] = segment_data
                
        except Exception as e:
            print(f"載入片段 {idx} 失敗: {str(e)}")
    
    def update_display(self):
        """更新顯示"""
        if self.current_file is None:
            return
            
        segment = self.segment_slider.value()
        self.segment_label.setText(f"Segment: {segment}")
        
        # 確保數據已載入
        if segment not in self.data_cache:
            self.load_segment(segment)
        
        if segment not in self.data_cache:
            return
            
        # 清除所有圖表
        self.ecg_plot.clear()
        self.ppg_plot.clear()
        self.ppg_first_diff_plot.clear()
        self.ppg_second_diff_plot.clear()
        self.abp_plot.clear()
        
        # 繪製信號
        data = self.data_cache[segment]
        x = np.arange(1250) / 125.0  # 轉換為時間軸（秒）
        
        # 繪製 ECG
        if self.checkboxes['ecg'].isChecked() and 'ecg' in data:
            self.ecg_plot.plot(x, data['ecg'], 
                             pen=pg.mkPen(color=(0,255,0), width=1))
            # 添加 R 波標記
            if 'annotations' in data:
                r_peaks = np.where(data['annotations'][:, 0] == 1)[0]
                r_peak_values = data['ecg'][r_peaks]
                self.ecg_plot.plot(x[r_peaks], r_peak_values, 
                                 pen=None, symbol='o', symbolSize=5, 
                                 symbolBrush=(255,0,0))
        
        # 繪製 PPG 及其差分
        if self.checkboxes['ppg'].isChecked() and 'ppg' in data:
            # 原始 PPG 信號
            ppg_signal = data['ppg']
            self.ppg_plot.plot(x, ppg_signal, 
                             pen=pg.mkPen(color=(255,0,0), width=1))
            
            # 計算並繪製一階差分
            ppg_first_diff = np.zeros_like(ppg_signal)
            # 內部點使用中心差分
            ppg_first_diff[1:-1] = (ppg_signal[2:] - ppg_signal[:-2]) / 2
            # 邊界點使用前向/後向差分
            ppg_first_diff[0] = ppg_signal[1] - ppg_signal[0]
            ppg_first_diff[-1] = ppg_signal[-1] - ppg_signal[-2]
            
            self.ppg_first_diff_plot.plot(x, ppg_first_diff,
                                        pen=pg.mkPen(color=(0,0,180), width=2))
            
            # 計算並繪製二階差分
            ppg_second_diff = np.zeros_like(ppg_signal)
            # 內部點使用中心差分
            ppg_second_diff[2:-2] = (ppg_signal[4:] - 2*ppg_signal[2:-2] + ppg_signal[:-4]) / 4
            # 邊界點使用相鄰的內部點值
            ppg_second_diff[0] = ppg_second_diff[2]
            ppg_second_diff[1] = ppg_second_diff[2]
            ppg_second_diff[-2] = ppg_second_diff[-3]
            ppg_second_diff[-1] = ppg_second_diff[-3]
            
            self.ppg_second_diff_plot.plot(x, ppg_second_diff,
                                         pen=pg.mkPen(color=(128,0,128), width=2))
            
            # 添加 PPG 特徵點
            if 'annotations' in data:
                # PPG peaks
                ppg_peaks = np.where(data['annotations'][:, 1] == 1)[0]
                ppg_peak_values = ppg_signal[ppg_peaks]
                self.ppg_plot.plot(x[ppg_peaks], ppg_peak_values, 
                                 pen=None, symbol='o', symbolSize=5, 
                                 symbolBrush=(0,255,0))
                
                # PPG turns
                ppg_turns = np.where(data['annotations'][:, 2] == 1)[0]
                ppg_turn_values = ppg_signal[ppg_turns]
                self.ppg_plot.plot(x[ppg_turns], ppg_turn_values,
                                 pen=None, symbol='s', symbolSize=5,
                                 symbolBrush=(255,128,0))
            
            # 添加圖例
            self.ppg_plot.addLegend()
        
        # 繪製 ABP
        if self.checkboxes['abp'].isChecked() and 'abp' in data:
            self.abp_plot.plot(x, data['abp'], 
                             pen=pg.mkPen(color=(0,0,255), width=1))
            
            # 添加 SBP/DBP 標記
            self.abp_plot.addLine(y=data['segsbp'], pen=pg.mkPen('r', style=Qt.DashLine))
            self.abp_plot.addLine(y=data['segdbp'], pen=pg.mkPen('b', style=Qt.DashLine))
        
        # 設置 X 軸標籤
        for plot in [self.ecg_plot, self.ppg_plot, self.ppg_first_diff_plot, 
                    self.ppg_second_diff_plot, self.abp_plot]:
            plot.setLabel('bottom', 'Time (s)')
        
        # 更新分析文本
        self.update_analysis_text(segment)
    
    def update_analysis_text(self, segment):
        """更新分析結果文本"""
        if segment not in self.data_cache:
            return
            
        data = self.data_cache[segment]
        text = f"Segment {segment} 分析結果\n"
        text += "=" * 30 + "\n\n"
        
        # 血壓值
        text += f"收縮壓 (SBP): {data['segsbp']:.1f} mmHg\n"
        text += f"舒張壓 (DBP): {data['segdbp']:.1f} mmHg\n"
        text += f"平均動脈壓 (MAP): {(data['segsbp'] + 2*data['segdbp'])/3:.1f} mmHg\n\n"
        
        # 個人信息
        info = data['personal_info']
        text += f"個人信息:\n"
        text += f"年齡: {info[0]:.0f}\n"
        text += f"性別: {info[1]:.0f}\n"
        text += f"體重: {info[2]:.1f} kg\n"
        text += f"身高: {info[3]:.1f} cm\n\n"
        
        # 血管特性
        vasc = data['vascular_properties']
        text += f"血管特性:\n"
        text += f"PTT: {vasc[0]:.2f} ms\n"
        text += f"PAT: {vasc[1]:.2f} ms\n"
        
        self.analysis_text.setText(text)
    
    def jump_to_segment(self, value):
        """跳轉到指定片段"""
        self.segment_slider.setValue(value)

def main():
    app = QApplication(sys.argv)
    viewer = H5DataViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 