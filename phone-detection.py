from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0, 720],
    [1250, 720],
    [1280, 0]
])

#Setting webcam resolution
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--webcam-resolution",
        type=int,
        default=[1280, 720],
        nargs=2
    )
    return parser.parse_args()

#Main function
def phone_detection():
    #Setting up webcam resolution
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    #Setting up video capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    #Setup model
    model = YOLO("yolo11n.pt")

    #Box setup
    box_annotator = sv.BoxAnnotator(thickness=2)

    #Analysis zones
    zone = sv.PolygonZone(polygon=ZONE_POLYGON)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

    while True:
        ret, frame = capture.read()

        #Analysing the image to see if a phone is present
        result = model(frame, classes=[67])[0]
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int)
        )

        #If a phone is detected, the counter will increase
        if len(detections) > 0:
            consecutive_detections += 1
        else:
            consecutive_detections = 0

        #Displaying the counter
        counter_text = f"Consecutive Frames: {consecutive_detections}"
        cv2.putText(
            frame, 
            counter_text, 
            (50, 50),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Scale
            (0, 255, 0) if consecutive_detections > 0 else (0, 0, 255),  # Color (green if detected, red if not)
            2  # Thickness
        )

        #If the counter reaches 15, the program will display a get off phone message
        if(consecutive_detections > 15):
            cv2.putText(
            frame, 
            "GET OFF PHONE", 
            (30, 200),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            5,  # Scale
            (0, 0, 255),  # Color (red)
            5  # Thickness
        )

        #Prepping the frame to be shown
        frame = box_annotator.annotate(scene=frame, detections=detections)   
        
        #Showing the frame
        cv2.imshow("Phone Detection", frame)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        #Adding a way to quit the program
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    #Cleanups
    capture.release()
    cv2.destroyAllWindows()


phone_detection()