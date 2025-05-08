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

        self.model = YOLO("/home/sungwoo/Workspace/OCRVision/OCRVision_best.pt")

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

        # print(num)
        
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