from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

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

def pt_from_lm(lm):
  return np.array([lm.x, lm.y, lm.z])

def get_face_angle(face_landmarks):
    pt1 = pt_from_lm(face_landmarks.landmark[1])
    pt2 = pt_from_lm(face_landmarks.landmark[2])
    pt3 = pt_from_lm(face_landmarks.landmark[168])

    forward_vector = pt1 - (pt2 + (pt3 - pt2) / 4.5)
    forward_vector /= np.linalg.norm(forward_vector)
    
    # angle of face from xz plane (0 is looking straight ahead, positive is looking upwards)
    if forward_vector[2] == 0: forward_vector[2] = 0.0001
    return np.arctan(forward_vector[1]/ forward_vector[2])


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

    consecutive_detections = 0

    fps = capture.get(cv2.CAP_PROP_FPS)
    face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    index = 0
    pool_xza = np.zeros(int(fps * 3))
    current_premium = 500

    while True:
        ret, frame = capture.read()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                angle_from_xz =  get_face_angle(face_landmarks)

                # calculate running avg of the angle from the xz plane
                pool_xza[index] = angle_from_xz
                index += 1
                index %= len(pool_xza)
                mean_angle = np.mean(pool_xza)
                if np.all(pool_xza != 0) and mean_angle < -0.55:
                    current_premium += 2
                    cv2.putText(
                        frame, 
                        "LOOK UP " + str(mean_angle), 
                        (30, 700),  # Position
                        cv2.FONT_HERSHEY_SIMPLEX,  # Font
                        5,  # Scale
                        (0, 0, 255),  # Color (red)
                        5  # Thickness
                    )

        #Analysing the image to see if a phone is present
        result = model(frame, classes=[67])[0]
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int)
        )

        #If a phone is detected, the counter will increase
        if len(detections) > 0:
            current_premium += 2
            consecutive_detections +=1
        else:
            current_premium -= 1
            consecutive_detections = 0

        #Displaying the counter
        counter_text = f"Current Preimium: {current_premium}"
        cv2.putText(
            frame, 
            counter_text, 
            (50, 50),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Scale
            (0, 255, 0),  # Color (green if detected, red if not)
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