# OCRVision 개발
## Setup
* 기존 ChemVision 프로젝트와 같이 라즈베리파이에 개발진행
* 5/6 패치로 libcamera 버전업으로 기존 코드가 작동이 안됨
	* Conda 방식은 Raspi 업그레이드에 취약함 > venv 방식으로 변경
* https://blockdmask.tistory.com/466
* https://blockdmask.tistory.com/566
* https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list
  
```bash
# 0.4.0미만 버전 확인 할 것!
# 19/11/2024 버전 사용 / libcamera=v0.3.2+99-1230f78d
libcamera-hello --version
rpicam-hello --list-cameras

sudo apt update 

# 환경 활성화
python -m venv ocr --system-site-packages
source ocr/bin/activate

# pip package 설치
pip install gradio
sudo apt update 
sudo apt install python3-pip -y 
pip install -U pip
pip install ultralytics
```
## Dataset Develop
* Yolov12n 활용
* 300batch 100epoch 세팅 / ncnn은 불필요하여 사용안함
	![[training.png]]
* https://app.roboflow.com/ds/7iwEs33L4b?key=qYNgHuofnR
* https://app.roboflow.com/ousiondeeplearning/digit-2xls6/1
* 라즈베리파이에서 추론시간 450ms 정도
![[2025-05-08 213435.png]](https://github.com/coport-uni/OCRVision_peal/blob/main/2025-05-08%20213435.png)
![[2025-05-08 213435.png]](https://github.com/coport-uni/OCRVision_peal/blob/main/2025-05-08%20213511.png)
![[confusion_matrix_normalized.png]](https://github.com/coport-uni/OCRVision_peal/blob/main/confusion_matrix_normalized.png)
## Code
```python title:yolo_trainer
from ultralytics import YOLO

class YoloTrainer():
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def trainer(self):

        self.model.train(
            data="/workspace/YOLO/version1/data.yaml",  # Path to dataset configuration file
            epochs=100,  # Number of training epochs
            batch=300,
            device="[0,1]",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        )

        self.model.val()

    def inferener(self):
        print("a")

    def exporter(self):
        self.model = YOLO("/workspace/YOLO/runs/detect/trainv11/weights/best.pt")
        self.model.export(format="ncnn")
        print("complete")

def main():
    yt=YoloTrainer()
    yt.exporter()
    # yt.trainer()

if __name__ == "__main__":
    main()

```

```python title:OCRVision
from picamera2 import Picamera2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import cv2
import time
import os


class OCRVision():
    def __init__(self):
        self.picam = Picamera2()
        self.config = self.picam.create_still_configuration({"size":(640,480),'format': 'RGB888'})
        self.picam.start(self.config)
        self.counter = 0

        self.model = YOLO("/home/sungwoo/Workspace/OCRVision/best.pt")

    def cameraRGB(self):
        output = self.picam.capture_image()
        output_a = np.array(output)
        text = "판독결과"

        return output, text
        # cv2.imwrite("test.jpg", output)

    def dataset_runner(self):
        output = self.picam.capture_image()
        time.sleep(1)
        self.picam.switch_mode_and_capture_file(self.config, f"/home/sungwoo/Workspace/OCRVision/dataset_raw/image_{self.counter}.jpg")
        self.counter = self.counter + 1

        return(output, self.counter)
        
    def yolo_detection(self):
        
        output = self.picam.capture_array()
        results = self.model(output)
        annotated_frame = results[0].plot()
        cls_array = []
        box_array = []

        for r in results:
            for i in range(len(r)):
                # print(int(r.boxes.cls[i]))
                # print(float(r.boxes.xyxyn[i][0]))
                cls_array.append(int(r.boxes.cls[i]))
                box_array.append(float(r.boxes.xyxyn[i][0]))
        
        data = [cls_array for _,cls_array in sorted(zip(box_array,cls_array))]
        # print(data)

        num = 0
        for i in range(len(data)):
            num += data[i] * 10**((len(data) - 2 - i))

        print(num)


        return annotated_frame, num

    def gui_launcher(self):
        with gr.Blocks() as demo:
            gr.Markdown("OCRVisionDemo-vision")
            output_img = gr.Image()
            vtxt = gr.Textbox(value="",label="촬영한 사진")
            vbutton = gr.Button(value="OCR 판독")
            vbutton.click(self.yolo_detection, inputs=[], outputs=[output_img, vtxt])

            gr.Markdown("OCRVisionDemo-dataset")
            dataset_img = gr.Image()
            dtxt = gr.Textbox(value="",label="촬영한 사진")
            dbutton = gr.Button(value="데이터세트 제작")
            dbutton.click(self.dataset_runner, inputs=[], outputs=[dataset_img, dtxt])

        demo.launch(share=True)

def main():

    ov = OCRVision()
    # ov.yolo_detection()
    ov.gui_launcher()

    # try: 
    #     ov = OCRVision()
    #     ov.gui_launcher()

    # except Exception as e:
    #     print(f'ERROR OCCURED: {e}')

if __name__ == "__main__":
    main()
```
## Result
![[confusion_matrix_normalized.png]](https://github.com/coport-uni/OCRVision_peal/blob/main/confusion_matrix_normalized.png)
![[confusion_matrix_normalized.png]](https://github.com/coport-uni/OCRVision_peal/blob/main/KakaoTalk_20250508_214300161.jpg)
