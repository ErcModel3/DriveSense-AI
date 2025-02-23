from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Run inference on the source and show results
results = model(source=0, show=True, conf=0.4,save=True, classes = [67])