import sys
import os
import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QCheckBox, QComboBox, QSlider, QLabel, QTextEdit)
from PyQt5.QtCore import Qt
from model_abp_1250points import ConvAutoencoder, ConvAutoencoder2, predict_reconstructed_1250abp
from model_pulse_representation import predict_reconstructed_abp
from abp_anomaly_detection_algorithms import detect_anomaly_gui, calculate_dtw_scores
from load_normap_ABP_data import load_training_set
import pickle
import torch
from scipy.ndimage import gaussian_filter1d

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.calc_button.clicked.connect(self.calculate_anomaly_score)
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
        self.checkboxes = {}
        for signal in ['ABP_Raw', 'ABP_F', 'ABP_SPeaks', 'ABP_Turns', 'ECG_Raw', 'ECG_F', 'ECG_RPeaks', 'PPG_Raw', 'PPG_F', 'PPG_SPeaks', 'PPG_Turns', 'Reconstructed_ABP']:
            self.checkboxes[signal] = QCheckBox(signal)
            self.checkboxes[signal].setChecked(True) if 'ABP' in signal else self.checkboxes[signal].setChecked(False)
            self.checkboxes[signal].stateChanged.connect(self.update_plot)
            left_layout.addWidget(self.checkboxes[signal])

        # 元信息顯示區域
        self.meta_info = QTextEdit()
        self.meta_info.setReadOnly(True)
        left_layout.addWidget(self.meta_info)

        # 右側面板 - 繪圖區域
        self.plot_widget = pg.PlotWidget(viewBox=CustomViewBox())
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Sample Points')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setClipToView(True)
        self.plot_widget.showGrid(x=True, y=True)

        top_layout.addWidget(left_panel, 1)
        top_layout.addWidget(self.plot_widget, 7)

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

        self.main_layout.addLayout(bottom_layout)

        self.data = {}
        self.current_file_path = ""

        self.load_files()

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
                else:
                    self.data[key] = [f[ref][:] for ref in matdata[key][0]]

        self.segment_slider.setMaximum(len(self.data['ABP_Raw']) - 1)
        self.update_meta_info()
        self.update_plot()
        # 在加载数据后初始化重构误差图
        if 'ABP_Raw' in self.data and len(self.data['ABP_Raw']) > 0:
            self.plot_reconstruction_error(self.data['ABP_Raw'][0])
        else:
            print("Warning: No ABP_Raw data found in the file.")


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

        # 设置y轴范围，留出一些边距
        y_min, y_max = min(y), max(y)
        y_range = y_max - y_min
        self.reconstruction_error_plot.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)


    def update_plot(self):
        self.plot_widget.clear()
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
                dtws = calculate_dtw_scores(y, turns)
                self.update_dtw_plot(dtws, turns)

        else:
            if not self.data:
                return

            x = np.arange(1250)  # 10 seconds at 125 Hz
            colors = {'ABP': (255, 0, 0), 'ECG': (0, 255, 0), 'PPG': (0, 0, 255)}

            for signal_type in ['ABP', 'ECG', 'PPG']:
                if self.checkboxes[f'{signal_type}_Raw'].isChecked():
                    y = self.data[f'{signal_type}_Raw'][segment]
                    self.plot_widget.plot(x, y, pen=pg.mkPen(color=colors[signal_type], width=2, name=f'{signal_type}_Raw'))

                    peak_key = 'ECG_RPeaks' if signal_type == 'ECG' else f'{signal_type}_SPeaks'
                    if self.checkboxes[peak_key].isChecked():
                        peaks = self.data[peak_key][segment]
                        peaks = peaks[peaks < 1250]
                        y_peaks = y[peaks]
                        self.plot_widget.plot(peaks, y_peaks, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=colors[signal_type])

                    if signal_type in ['ABP', 'PPG'] and self.checkboxes[f'{signal_type}_Turns'].isChecked():
                        turns = self.data[f'{signal_type}_Turns'][segment]
                        turns = turns[turns < 1250]
                        y_turns = y[turns]
                        self.plot_widget.plot(turns, y_turns, pen=None, symbol='s', symbolPen=None, symbolSize=5, symbolBrush=colors[signal_type])

                if self.checkboxes[f'{signal_type}_F'].isChecked():
                    y = self.data[f'{signal_type}_F'][segment]
                    self.plot_widget.plot(x, y, pen=pg.mkPen(color=colors[signal_type], width=2, name=f'{signal_type}_F'))# style=Qt.DashLine, 

            if 'ABP_Raw' in self.data and 'ABP_Turns' in self.data and 'ABP_SPeaks' in self.data:
                abp_data = self.data['ABP_Raw'][segment]
                # 绘制高斯平滑后的ABP
                if self.checkbox_smoothed.isChecked():
                    smoothed_abp = gaussian_filter1d(abp_data, sigma=2)
                    self.plot_widget.plot(x, smoothed_abp, pen=pg.mkPen(color=(100, 155, 155), width=2, name='Smoothed ABP'))
                    # 绘制二阶导数
                    if self.checkbox_second_derivative.isChecked():
                        second_derivative = np.gradient(np.gradient(smoothed_abp))
                        self.plot_widget.plot(x, second_derivative, pen=pg.mkPen(color=(255, 0, 255), width=2, name='ABP Second Derivative'))



                turns = self.data['ABP_Turns'][segment]
                speaks = self.data['ABP_SPeaks'][segment]
                # anomaly_score = detect_anomaly_gui(data, turns, speaks)
                # self.anomaly_score_label.setText(f"Anomaly Score: {anomaly_score:.4f}")
                segment = self.segment_slider.value()
                abp_data = self.data['ABP_Raw'][segment]
                turns = self.data['ABP_Turns'][segment]
                turns = turns[turns < 1250]  # 确保turns在有效范围内
                dtws = calculate_dtw_scores(abp_data, turns)

                # 更新 meta_info 中的 DTW 分数
                dtw_text = 'DTW Scores: ['
                for dtw in dtws[:-1]:
                    dtw_text += f"{dtw:.4f}, "
                dtw_text += f"{dtws[-1]:.4f}]"
                
                # 更新 meta_info，只保留 DTW 分数和原有的元数据
                current_text = self.meta_info.toPlainText()
                updated_text = dtw_text + "\n" + "\n".join(current_text.split("\n")[1:])
                self.meta_info.setText(updated_text)
                
                # 更新 DTW 图
                self.update_dtw_plot(dtws, turns)
            else:
                self.anomaly_score_label.setText("Anomaly Score: N/A")

            # segment = self.segment_slider.value()
            # if self.checkboxes['Reconstructed_ABP'].isChecked() and 'ABP_Raw' in self.data and 'ABP_Turns' in self.data:
            #     # reconstructed_abp = predict_reconstructed_abp(self.data['ABP_Raw'][segment], self.data['ABP_Turns'][segment])
            #     data = self.data['ABP_Raw'][segment]
            #     reconstructed_abp, self.reconstructed_error = predict_reconstructed_1250abp(data)
            #     self.plot_widget.plot(x, reconstructed_abp, pen=pg.mkPen(color=(0, 0, 0), width=2, name='Reconstructed_ABP'))    
            #     self.plot_reconstruction_error(data)

        self.plot_widget.autoRange()


    def calculate_anomaly_score(self):
        if self.db_combo.currentText() != "training_set.npz" and 'ABP_Raw' in self.data and 'ABP_Turns' in self.data and 'ABP_SPeaks' in self.data:
            segment = self.segment_slider.value()
            data = self.data['ABP_Raw'][segment]
            turns = self.data['ABP_Turns'][segment]
            speaks = self.data['ABP_SPeaks'][segment]
            anomaly_score = detect_anomaly_gui(data, turns, speaks)
            self.anomaly_score_label.setText(f"Anomaly Score: {anomaly_score:.4f}")
        else:
            self.anomaly_score_label.setText("Anomaly Score: N/A")

    def plot_reconstruction_error(self, data):
        self.reconstruction_error_plot.clear()

        x = np.arange(len(self.reconstructed_error))
        self.reconstruction_error_plot.plot(x, self.reconstructed_error, pen=pg.mkPen(color=(255, 0, 0), width=2))
        self.reconstruction_error_plot.setYRange(0, np.max(self.reconstructed_error) * 1.1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PulseDBViewer()
    window.show()
    sys.exit(app.exec_())