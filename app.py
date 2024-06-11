from flask import Flask, render_template, Response
import torch
import numpy as np
import cv2
from picamera2 import Picamera2
import time
import threading
from playsound import playsound

hp = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/pi/project/smodel2.pt', force_reload=True)

cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"size": (1920, 1080)}))
cam.start()
time.sleep(2)

detect_result = []

def capture(cam):
    pic = cam.capture_array()
    return pic

def smoke_detect(pic, model):
    result = model(pic)
    return result

def detect_thread():
    global detect_result
    while True:
        pic = capture(cam)
        detect_result = smoke_detect(pic, model).xyxy[0].cpu().numpy()

def make_video():
    global detect_result
    while True:
        frm = capture(cam)
        for result in detect_result:
            xmin, ymin, xmax, ymax, conf, cls = result
            if conf > 0.3:
                cv2.rectangle(frm, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        ret, buf = cv2.imencode('.jpg', frm)
        frm = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frm + b'\r\n')

@hp.route('/video')
def video():
    return Response(make_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@hp.route('/play_sound', methods=['POST'])
def play_sound():
    playsound('/home/pi/project/templates/warning.mp3')
    return '', 204
    
@hp.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    threading.Thread(target=detect_thread).start()
    hp.run(host='0.0.0.0', port=5000)
