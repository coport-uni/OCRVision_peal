from ultralytics import YOLO

class YoloTrainer():
    def __init__(self):
        self.model = YOLO("yolo12n.pt")

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
