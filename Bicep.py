import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculateAngle(a ,b , c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# count variables
counter = 0
stage = None
form = None

capture = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

    while capture.isOpened():
        ret, frame  = capture.read()

        # convert to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False


        results = pose.process(image)

        # convert to bgr
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark

            # get co-ordinates
            shoulder = [landmarks[11].x, landmarks[11].y] 
            elbow = [landmarks[13].x, landmarks[13].y] 
            wrist = [landmarks[15].x, landmarks[15].y]

            angle = calculateAngle(shoulder, elbow, wrist)

            left_waist = [landmarks[23].x, landmarks[23].y]
            left_knee = [landmarks[25].x, landmarks[25].y]

            back_angle = calculateAngle(shoulder, left_waist, left_knee)
 
            # visualize angles

            cv2.putText(image, str(angle), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, str(back_angle), 
                        tuple(np.multiply(left_waist, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            

            if angle > 160:
                stage = 'down'

            if angle < 40 and stage == 'down':
                stage = 'up'
                counter += 1
                print(counter)
            
            if  back_angle < 175:
                form = 'bad'
            else:
                form = 'good'
                

                        
             

        except :
            pass

        # render curl counter
        cv2.rectangle(image, (0, 0), (225, 73), (240, 120 , 20), -1)

        cv2.putText(image, str(counter),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, form,
                (140, 40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, "Reps ",
                (50, 40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,255), 2, cv2.LINE_AA)

        # render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS  )

        cv2.imshow('Mediapipe',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()