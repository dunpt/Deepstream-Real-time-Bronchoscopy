from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import *#QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QMainWindow, QStatusBar, QDesktopWidget, QMessageBox, QFrame
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QTimer
import sys
import cv2
import numpy as np
from PyQt5.QtMultimedia import QCameraInfo
import os
import shutil
import sys, time
import subprocess
import json
import pygame

#/var/log/syslog

pygame.mixer.pre_init(frequency=48000, buffer=2048)
pygame.mixer.init()
detected_sound = pygame.mixer.Sound('/home/ailab/Documents/kcutp_ver4/pip_sound.mp3')

os.system('Xvfb :1 -screen 0 1600x1200x16  &')    # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.environ['DISPLAY']=':1.0'    # tell X clients to use our virtual DISPLAY :1.0
os.environ['QT_DEBUG_PLUGINS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/ailab/.miniconda3/envs/myenv/lib/python3.10/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_PLUGIN_PATH'] = '/home/ailab/.miniconda3/envs/myenv/lib/python3.10/site-packages/PyQt5/Qt5/plugins'

container_id = '2d4e7ef33a3d'
command_to_run = "python3 /opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/main_back_to_back_NB_test.py -i /opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/segment_1.h264"

start_container_command = ["docker", "start", container_id]

docker_exec_command = [
    "docker", "exec", 
    "-d", 
    container_id, 
    "sh", "-c", command_to_run
]

kill_command = [
    "docker", "exec",
    container_id,
    "pkill", "-f", command_to_run  
]

unmute_command = ["amixer", "-c", "0", "set", "Master", "playback", "50%", "unmute"]

mute_command = ["amixer", "-c", "0", "set", "Master", "playback", "1%", "unmute"]

def copy_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp')):
            src_file = os.path.join(source_folder, filename)
            dst_file = os.path.join(destination_folder, filename)
            
            shutil.copy(src_file, dst_file)

def overlay_mask_edges(image_path, mask_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.resize(image, (800, 800))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found at {mask_path}")
    mask = cv2.resize(mask, (800, 800))

    edges = cv2.Canny(mask, threshold1=100, threshold2=200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_colored[edges == 255] = [255, 255, 0]
    overlay_image = cv2.addWeighted(image, 1, edges_colored, 0.5, 0)

    return overlay_image

class RTSPThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    new_image = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.out = None

        self.capture = False

        self.image_count = 0
        
        self.known_files = set()

        

    def run(self):
        self._run_flag = True

        self.cap = cv2.VideoCapture('rtsp://localhost:8555/ds-test')

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out_path = os.path.join(os.getcwd(), 'output.mp4')

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

        frame_path = os.path.join(os.getcwd(), 'output.png')
    
        # self.cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                self.out.write(cv_img)
                if self.capture:
                    cv2.imwrite(frame_path, cv_img)
                    self.capture = False
                
                # file_path = '/home/ailab/Documents/kcutp_ver4/connect_docker/transfer.json'
                file_path = '/home/ailab/Documents/kcutp_ver4/connect_docker/transfer.json'
                with open(file_path, 'r') as file:
                    data = json.load(file)

                folder_name = data['PatientID']
                patient_folder = os.path.join(os.getcwd(), "connect_docker", f'{folder_name}', "imgs",)
                os.makedirs(patient_folder, exist_ok=True)

                current_files = set([f for f in os.listdir(patient_folder) if f.endswith('.jpg')])
                new_files = current_files - self.known_files  # Find newly added files

                if new_files:
                    for new_file in sorted(new_files):  # Sort the files for better handling (if needed)
                        new_image_path = os.path.join(patient_folder, new_file)
                        mask_path = new_image_path.replace('imgs', 'mask')

                        save_dir  = f'Saved_data/{folder_name}/AI_chan_doan'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        cv_img = overlay_mask_edges(new_image_path, mask_path)
                        image_path = os.path.join(save_dir, f'image_{self.image_count}.jpg')
                        cv2.imwrite(image_path, cv_img)
                        self.new_image.emit(image_path)
                        play_time_ms = 3000 
                        detected_sound.play(maxtime=play_time_ms)
                        self.image_count += 1

                self.known_files = current_files

           
                
        self.cap.release()
        self.out.release()
    
    def capture_frame(self):
        self.capture = True

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class NormalThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    new_image = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.out = None

        self.image_count = 0
        self.frame_counter = 0
        save_dir = 'captured_images'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def run(self):
        # Capture from the camera
        self.cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out_path = os.path.join(os.getcwd(), 'output.mp4')
        # print("out_path", out_path)

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                self.out.write(cv_img)
                self.frame_counter += 1  

                if self.frame_counter == 50:

                    image_path = os.path.join(save_dir, f'image_{self.image_count}.jpg')
                    cv2.imwrite(image_path, cv_img)
                    
                    self.new_image.emit(image_path)
                    
                    self.image_count += 1
                    self.frame_counter = 0

        self.cap.release()
        self.out.release()


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class CaptureDialog(QDialog):
    def __init__(self, parent=None):
        super(CaptureDialog, self).__init__(parent)
        self.setWindowTitle("Capture Image Confirmation")
        uic.loadUi('captureimg_popup.ui', self)
        self.show()

        output_path = os.path.join(os.getcwd(), 'output.png')
        self.img = cv2.imread(output_path)

        resized_frame = cv2.resize(self.img, (521, 491))  

        height, width, channel = resized_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        self.label.setPixmap(pixmap)

        self.pushButton.clicked.connect(self.accept)
        self.pushButton_2.clicked.connect(self.reject)

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.available_cameras = QCameraInfo.availableCameras()  # Getting available cameras
        # self.setFixedSize(920, 1020)

        uic.loadUi('ui_ver3.ui', self)
        self.show()
        self.video_mode = 'webcam'
        self.flag = True

        self.ss_video.setEnabled(False)
        self.ss_video_2.setEnabled(False)
        # self.ss_video_3.setEnabled(False)
        

        self.current_folder = os.getcwd()
        self.image_index = 0
        self.video_save_path = ''
        self.ss_video_4.clicked.connect(self.SetPatientID)
        self.ss_video.clicked.connect(self.ClickStartVideo)
        self.ss_video_2.clicked.connect(self.CaptureImage)
        self.ss_video_3.clicked.connect(self.RefreshBool)

        self.ss_video_3.setEnabled(False)

        # image_paths = ['/Users/dungpt1504/Downloads/100_ca_dataset/BH0308/SVCD_085451_0.mpg_snapshot_08.06.599.jpg', 
        #            '/Users/dungpt1504/Downloads/100_ca_dataset/BH0308/SVCD_085451_0.mpg_snapshot_06.26.375.jpg', 
        #            '/Users/dungpt1504/Downloads/100_ca_dataset/BH0308/SVCD_085451_0.mpg_snapshot_05.22.569.jpg']  
        
        
        # for image_path in image_paths:
        #     label = QLabel(self)
        #     pixmap = QPixmap(image_path)
        #     label.setPixmap(pixmap)
        #     label.setAlignment(Qt.AlignCenter)
        #     self.imageLayout.addWidget(label)

        # self.scrollArea.setWidgetResizable(True) 
    
    
        
    def RefreshBool(self):
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")
        self.lineEdit_3.setText("")
        self.lineEdit_4.setText("")
        self.lineEdit_5.setText("")
        self.lineEdit_6.setText("")
        self.ss_video.setEnabled(False)
        self.ss_video_2.setEnabled(False)

        pixmap = QPixmap(881,881)
        pixmap.fill(Qt.white)
        self.image_label.setPixmap(pixmap)

        
        
        self.statusbar.showMessage('Nhập thông tin bênh nhận mới')

        for i in reversed(range(self.imageLayout.count())): 
            self.imageLayout.itemAt(i).widget().setParent(None)

        message = QMessageBox()
        message.setText('Tạo ca mới thành công')
        message.exec_()

        


    def CaptureImage(self):
        if self.thread and self.thread.isRunning():
            self.thread.capture_frame()
            time.sleep(0.2)
            # if ret:
            dialog = CaptureDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                self.flag = True
            else:
                self.flag = False
            # else:
            #     self.statusbar.showMessage('Lỗi không chụp được ảnh')

            flag_string = str(self.flag)
            patient_folder = os.path.join(self.current_folder, "Saved_data", self.lineEdit_4.text(), "Images", flag_string)
            os.makedirs(patient_folder, exist_ok=True)
            path_save = os.path.join(patient_folder, f"{self.image_index}.png")

            self.image_index += 1
            # cv2.imwrite(path_save, frame)
            shutil.copyfile(os.path.join(os.getcwd(), "output.png"), path_save)
            self.statusbar.showMessage('Ảnh được chụp và lưu ở ' + path_save)
            os.remove(os.path.join(os.getcwd(), "output.png"))


    def SetPatientID(self):
        if (self.lineEdit.text() != "" and self.lineEdit_2.text() != "" and 
            self.lineEdit_3.text() != "" and self.lineEdit_4.text() != "" and 
            self.lineEdit_5.text() != ""):
            
            message = QMessageBox()
            combined_message = (
                'Họ và tên: ' + self.lineEdit.text() + '\n' +
                'Tuổi: ' + self.lineEdit_2.text() + '\n' +
                'Giới: ' + self.lineEdit_3.text() + '\n' +
                'Chẩn đoán: ' + self.lineEdit_5.text() + '\n' +
                'Mã bệnh nhân: ' + self.lineEdit_4.text() + '\n' +
                'Thời gian: ' + self.lineEdit_6.text()
            )
            message.setText(combined_message)
            message.exec_()

            data = {
                'Name': self.lineEdit.text(),
                'Age': self.lineEdit_2.text(),
                'Sex': self.lineEdit_3.text(),
                'Diagnose': self.lineEdit_5.text(),
                'PatientID': self.lineEdit_4.text(),
                'Time': self.lineEdit_6.text()
            }

            patient_folder = os.path.join(self.current_folder, "Saved_data", self.lineEdit_4.text())
            os.makedirs(patient_folder, exist_ok=True)
            json_savepath = os.path.join(patient_folder, f"{self.lineEdit_4.text()}.json")
            with open(json_savepath, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)

            self.ss_video.setEnabled(True)
            self.ss_video_2.setEnabled(True)
        else:
            self.ss_video.setEnabled(False)
            self.ss_video_2.setEnabled(False)
            


    def ClickStartVideo(self):
        
        # Change button color to light blue
        self.ss_video.clicked.disconnect(self.ClickStartVideo)
        self.statusbar.showMessage('Luồng streaming đang chạy...')
        # subprocess.run(unmute_command)
        # Change button to stop
        self.ss_video.setText('Kết thúc streaming')
        self.thread = RTSPThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.new_image.connect(self.update_image_view)

        # start the thread
        self.thread.start()
        # self.ss_video.clicked.connect(self.thread.stop)  # Stop the video if button clicked
        self.ss_video.clicked.connect(self.EndVideo)

        data = {
            'PatientID': self.lineEdit_4.text(),
            'bool':0,
            'bool_log':1,
            "show_image": 0,
            "time_skip":0
        }

        with open("/home/ailab/Documents/kcutp_ver4/connect_docker/transfer.json", 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)

        self.ss_video_3.setEnabled(False)
            

   

    def EndVideo(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.image_label.clear()

            original_video_path = os.path.join(os.getcwd(), 'output.mp4')
            os.makedirs(os.path.join(self.current_folder, "Saved_data", f"{self.lineEdit_4.text()}"), exist_ok=True)
            target_path = os.path.join(self.current_folder, "Saved_data", f"{self.lineEdit_4.text()}", f"{self.lineEdit_4.text()}.mp4")

            shutil.copyfile(original_video_path, target_path)
            os.remove(original_video_path)

            message = QMessageBox()
            message.setText('Streaming đã dừng và video được lưu ' + target_path)
            message.exec_()

            self.statusbar.showMessage('Streaming đã dừng và video được lưu ' + target_path)

            # subprocess.run(mute_command)

            data = {
                'PatientID': self.lineEdit_4.text(),
                'bool':0,
                'bool_log':0,
                "show_image": 0,
                "time_skip":0
            }

            with open("/home/ailab/Documents/kcutp_ver4/connect_docker/transfer.json", 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)

            destination_folder = os.path.join(os.getcwd(), "Saved_data", f'{self.lineEdit_4.text()}', "Raw_image")
            source_folder = os.path.join(os.getcwd(), "connect_docker", f'{self.lineEdit_4.text()}', "imgs")
            copy_images(source_folder, destination_folder)

            # self.statusbar.showMessage('Streaming đã dừng và video được lưu')
            self.ss_video.setText('Bắt đầu streaming')
            self.ss_video.clicked.disconnect(self.EndVideo)
            self.ss_video.clicked.connect(self.ClickStartVideo)
            self.ss_video_3.setEnabled(True)
        else:
            self.statusbar.showMessage('Không có video nào đang chạy')
           

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_image_view(self, image_path):
        label = QLabel(self)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        self.imageLayout.addWidget(label)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(881, 881, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spash Screen Example')
        self.setFixedSize(1100, 500)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.counter = 0
        self.n = 300 # total instance

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.loading)
        self.timer.start(30)
    
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.frame = QFrame()
        layout.addWidget(self.frame)

        self.labelTitle = QLabel(self.frame)
        self.labelTitle.setObjectName('LabelTitle')

        # center labels
        self.labelTitle.resize(self.width() - 10, 150)
        self.labelTitle.move(0, 40) # x, y
        self.labelTitle.setText('Ứng dụng Chẩn đoán UTP')
        self.labelTitle.setAlignment(Qt.AlignCenter)

        self.labelDescription = QLabel(self.frame)
        self.labelDescription.resize(self.width() - 10, 50)
        self.labelDescription.move(0, self.labelTitle.height())
        self.labelDescription.setObjectName('LabelDesc')
        self.labelDescription.setText('<strong>Working on Task #1</strong>')
        self.labelDescription.setAlignment(Qt.AlignCenter)

        self.progressBar = QProgressBar(self.frame)
        self.progressBar.resize(self.width() - 200 - 10, 50)
        self.progressBar.move(100, self.labelDescription.y() + 130)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFormat('%p%')
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, self.n)
        self.progressBar.setValue(20)

        self.labelLoading = QLabel(self.frame)
        self.labelLoading.resize(self.width() - 10, 50)
        self.labelLoading.move(0, self.progressBar.y() + 70)
        self.labelLoading.setObjectName('LabelLoading')
        self.labelLoading.setAlignment(Qt.AlignCenter)
        self.labelLoading.setText('loading...')

    def loading(self):
        self.progressBar.setValue(self.counter)

        if self.counter == int(self.n * 0.3):
            self.labelDescription.setText('<strong>Working on Task #2</strong>')
        elif self.counter == int(self.n * 0.6):
            self.labelDescription.setText('<strong>Working on Task #3</strong>')
        elif self.counter >= self.n:
            self.timer.stop()
            self.close()

            time.sleep(1)

            self.MyWindow = QtWidgets.QMainWindow()
            self.myApp = MyWindow()
            # self.myApp.setupUi(self.MyWindow)
            # self.MyWindow.show()

        self.counter += 1

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # win = MyWindow()
    # win.show() 
    # sys.exit(app.exec())
    

        #     #LabelTitle {
        #     font-size: 60px;
        #     color: #93deed;
        # }
        
        # #LabelDesc {
        #     font-size: 30px;
        #     color: #c2ced1;
        # }

        # #LabelLoading {
        #     font-size: 30px;
        #     color: #e8e8eb;
        # }

    
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet('''
        #LabelTitle {
            font-size: 60px;
            color: #FFFFFF;
        }
        
        #LabelDesc {
            font-size: 30px;
            color: #FFFFFF;
        }

        #LabelLoading {
            font-size: 30px;
            color: #e8e8eb;
        }

        QProgressBar {
            background-color: white;
            color: rgb(200, 200, 200);
            border-style: none;
            border-radius: 10px;
            text-align: center;
            font-size: 30px;
        }

        QProgressBar::chunk {
            border-radius: 10px;
            background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #1C3334, stop:1 #376E6F);
        }
    ''')

    # MyWindow = QtWidgets.QMainWindow()
    # ui = Ui_MyWindow()
    # ui.setupUi(MyWindow)
    # MyWindow.show()
    # process = subprocess.Popen(['./monitor_program.sh'])
    subprocess.run(start_container_command)
    subprocess.run(kill_command)
    subprocess.run(docker_exec_command)

    splash = SplashScreen()
    splash.show()
    # sys.exit(app.exec_())
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')
        # pid = process.pid

        # os.kill(pid, signal.SIGTERM)
        subprocess.run(kill_command)
