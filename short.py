import cv2
import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torchvision.transforms as T
import cv2
import os
import sys
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as T

from PIL import Image
from model import siamese_model
from facenet_pytorch import MTCNN, InceptionResnetV1
# Define the device
device = torch.device('cpu')
torch.cuda.empty_cache()
from PIL import Image
from model import siamese_model
from facenet_pytorch import MTCNN, InceptionResnetV1

# Define the device
device = torch.device('cpu')
torch.cuda.empty_cache()

# Excel file path
attendance_file = "attendance.xlsx"





# Function to check and update attendance
def update_attendance(name):
    date_today = time.strftime("%Y-%m-%d")
    time_now = time.strftime("%H:%M:%S")
    
    if not os.path.exists(attendance_file):
        # Create a new DataFrame and save it to the Excel file if it doesn't exist
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(attendance_file, index=False)

    # Read the Excel file
    df = pd.read_excel(attendance_file)

    # Check if the person is already marked present today
    if ((df["Name"] == name) & (df["Date"] == date_today)).any():
        return "You are already marked present"
    
    # Append the new attendance record
    new_record = {"Name": name, "Date": date_today, "Time": time_now}
    df = df.append(new_record, ignore_index=True)
    
    # Save the updated DataFrame to the Excel file
    df.to_excel(attendance_file, index=False)
    
    return "Attendance marked successfully"

def main():
    cooldown_limit = 0.5  # Minimum time needed for model to confirm change in number of people in frame
    regular_check_limit = 3  # Regular classification check
    db_path = "database/"
    siamese_model_path = "saved_models/siamese_model"
    load_from_file = True
    yolov5_type = "yolov5m"
    screen_size = (800, 600)
    scale = (1, 1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-db",
        "--db_path",
        help="Database path . Use relative path . Default path : " + db_path,
    )
    parser.add_argument(
        "-smp",
        "--siamese_model_path",
        help="Siamese Model path . Use relative path . Default path : "
        + siamese_model_path,
    )
    parser.add_argument(
        "-load",
        "--load_from_file",
        help="[TRUE] if you want to load reference embeddings from previously generated file , [FALSE] if you want to recompile or create new embeddings for the reference images . Default is set to TRUE",
    )
    parser.add_argument(
        "-yolov5",
        "--yolov5_type",
        help="Enter which yolov5 model you want to use : [yolov5s] ,[yolov5m] ,[yolov5l] ,[yolov5x] . Default type : yolov5m ",
    )
    parser.add_argument(
        "-cdl",
        "--cooldown_limit",
        help=f" Lower the cooldown higher the precision higher the memory usage . Default value : {cooldown_limit}s",
    )
    parser.add_argument(
        "-rcl",
        "--regular_check_limit",
        help=f" Helps in correcting previous errors by either the camera or the program . Default value : {regular_check_limit}s",
    )
    parser.add_argument(
        "-size",
        "--screen_size",
        help=f"Set Default screen size for the webcam feed : [SCREEN_W*SCREEN_H] . Default size : {screen_size[0]}*{screen_size[1]} ",
    )
    parser.add_argument(
        "-scale",
        "--scale",
        help=f"Set Default scale for the webcam feed : [SCALE_X*SCALE_Y] . Default size : {scale[0]}*{scale[1]} ",
    )

    args = parser.parse_args()
    if args.db_path:
        db_path = args.db_path

    if args.siamese_model_path:
        siamese_model_path = args.siamese_model_path

    if args.load_from_file:
        if args.load_from_file.upper() == "FALSE":
            load_from_file = False

    if args.yolov5_type:
        yolov5_type = args.yolov5_type

    if args.cooldown_limit:
        cooldown_limit = float(args.cooldown_limit)

    if args.regular_check_limit:
        regular_check_limit = float(args.regular_check_limit)

    if args.screen_size:
        screen_size = []
        for i in args.screen_size.split("*"):
            screen_size.append(int(i))

    if args.scale:
        scale = []
        for i in args.scale.split("*"):
            scale.append(int(i))

    # Initializing all the models and reference images
    device, classes, loader, reference_cropped_img, yolov5, resnet, mtcnn, model = init(
        load_from_file=load_from_file,
        db_path=db_path,
        siamese_model_path=siamese_model_path,
        yolov5_type=yolov5_type,
    )

    # Initializing cooldown clocks and Face-Recognition paramaters
    n_people = 0  # Number of people confirmed after cooldown
    cooldown = 0
    new_frame_time = 0
    prev_frame_time = 0
    regular_check_cooldown = 0

    classify_faces = True  # Wheather to use MTCNN to classify faces
    start_cooldown = False  # Starts when there is change in number of people
    person_names = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        new_frame_time = time.time()
        time_diff = new_frame_time - prev_frame_time
        fps = int(1 / (time_diff))
        regular_check_cooldown = regular_check_cooldown + time_diff
        fps = cap.get(cv2.CAP_PROP_FPS)

        boxes_info = yolov5(frame).xyxy[0].cpu().numpy().tolist()
        person_boxes = []  # selecting person class alone

        ith_n_people = 0  # number of person in frame at the moment (It might be wrong and is confirmed through cooldown)
        for i in boxes_info:
            if i[5] == 0:  # class for person is 0
                person_boxes.append(tuple(i[:4]))
                ith_n_people = ith_n_people + 1

        if ith_n_people != n_people and start_cooldown == False:
            start_cooldown = True

        if ith_n_people == n_people:
            cooldown = 0
            start_cooldown = False

        if regular_check_cooldown >= regular_check_limit:
            regular_check_cooldown = 0
            classify_faces = True

        if start_cooldown:
            cooldown = cooldown + time_diff

        if (
            cooldown >= cooldown_limit
        ):  # Confirming if the change in number is slight error
            n_people = ith_n_people
            cooldown = 0
            start_cooldown = False
            classify_faces = True  # Number of people in frame is changed so we feed the frame into MTCNN

        if (
            classify_faces and n_people == 0
        ):  # If number of peopel is 0 in frame then there is no need to classify
            person_names = []
            face_boxes = []
            face_name = []
            classify_faces = False

        if classify_faces:
            # Initializing new boxes and person name
            person_names = []
            face_boxes = []
            face_name = []

            classify_faces = False
            boxes, probs, points = mtcnn.detect(frame[:, :, ::-1], landmarks=True)

            if boxes is not None:
                for box in boxes:  # classifying predicted boxes
                    predicted_class, similarity = classify(
                        box,
                        frame,
                        loader,
                        resnet,
                        model,
                        reference_cropped_img,
                        classes,
                        device,
                    )
                    face_boxes.append(box)
                    if predicted_class == -1:
                        face_name.append("Stranger")
                    else:
                        face_name.append(predicted_class)

            for i in person_boxes:
                temp_name = ""
                new_max = 0
                for j, k in zip(face_boxes, face_name):
                    iou = IOU(
                        i, j, screen_size=tuple(frame.shape[:2])
                    )  # The box for the person and box for the person's face must intersect the highest
                    if new_max < iou:
                        new_max = iou
                        temp_name = k

                person_names.append(temp_name)
        
        check_repeat = (
            []
        )  # Make faces of same person cannot present at the same time (prevent error due to yolov5 smaller models)
        for i, j in zip(person_boxes, person_names):
            if j in check_repeat:
                continue
            check_repeat.append(j)
            x_min, y_min, x_max, y_max = i
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            bgr = (0, 255, 0)

            if j == "Stranger":
                bgr = (0, 0, 255)
                name = "Stranger"
            else:
                name = classes[j]

            label = f"{name}"
            thickness = 1
            t_size = cv2.getTextSize(label, 0, fontScale=thickness / 2, thickness=1)[0]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bgr, 2)

            cv2.putText(
                frame,
                label,
                (x_min, y_min - 10),
                0,
                thickness / 2,
                bgr,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            if name != "Stranger":
                # Update attendance and get the message
                message = update_attendance(name)
                print(message)  # Print the message (optional, you can also display it on the screen)
                cv2.putText(frame, message, (x_min, y_min - 25), 0, thickness / 2, bgr, thickness=1, lineType=cv2.LINE_AA)

        frame = cv2.resize(frame, screen_size)
        cv2.imshow("Webcam", frame)

        prev_frame_time = new_frame_time
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


def init(load_from_file=True, db_path="database/", siamese_model_path="saved_models/siamese_model", yolov5_type="yolov5m"):
    # Assuming necessary imports are already made
    from model import siamese_model
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    from yolov5 import YOLOv5
    
    # Load the YOLOv5 model
    yolov5 = YOLOv5(yolov5_type, device=device)
    
    # Load the InceptionResnetV1 model
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Load the MTCNN model
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Load the Siamese model
    model = siamese_model()
    model.load_state_dict(torch.load(siamese_model_path, map_location=device))
    model.eval()
    
    # Load reference images and their embeddings
    classes = []
    loader = None
    reference_cropped_img = []

    if load_from_file:
        # Load from previously generated file (example, might need adjustment)
        reference_cropped_img = torch.load(db_path + "reference_cropped_img.pt")
        classes = torch.load(db_path + "classes.pt")
    else:
        # Code to compile new embeddings (example, might need adjustment)
        # Assuming the loader and classes are set appropriately
        pass

    return device, classes, loader, reference_cropped_img, yolov5, resnet, mtcnn, model

if __name__ == "__main__":
    main()
