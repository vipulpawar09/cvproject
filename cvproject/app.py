from flask import Flask, render_template, request, redirect, url_for,jsonify
import cv2
import numpy as np
import time
import PoseModule as pm

app = Flask(__name__)
cap = cv2.VideoCapture(0)
@app.route('/')
def home():
    """Render the home page where users can select an exercise."""
    return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None
        return jsonify(success=True)
    return jsonify(success=False)


@app.route('/start', methods=['POST'])
def start_exercise():
    """Start the exercise tracking based on user selection."""
    exercise = request.form.get('exercise')  # Get the selected exercise from the form
    if exercise == "curls":
        track_curls()
    elif exercise == "pushups":
        track_hand_uplift()
    elif exercise == "squats":
        track_knee_raises()
    return redirect(url_for('home'))  # Return to home after exercise

def track_curls():
    """Tracking logic for bicep curls."""
    cap = cv2.VideoCapture(0)
    detector = pm.poseDetector()
    count, dir, pTime = 0, 0, 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to access the camera.")
            break

        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("Curl Exercise Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track_hand_uplift():
    """Tracking logic for hand uplift exercise with visual posture feedback."""
    cap = cv2.VideoCapture(0)
    detector = pm.poseDetector()
    count, pTime = 0, 0
    movement_completed = False

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to access the camera.")
            break

        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            # Track vertical hand position
            # Use landmarks:
            # 11 (left shoulder), 13 (left elbow), 15 (left wrist)
            # 12 (right shoulder), 14 (right elbow), 16 (right wrist)

            # Get y-coordinates for left and right body parts
            left_shoulder_y = lmList[11][2]
            right_shoulder_y = lmList[12][2]
            left_wrist_y = lmList[15][2]
            right_wrist_y = lmList[16][2]

            # Calculate vertical lift percentage
            left_lift = max(0, left_shoulder_y - left_wrist_y)
            right_lift = max(0, right_shoulder_y - right_wrist_y)

            # Map lift to percentage
            left_per = np.interp(left_lift, (0, 200), (0, 100))
            right_per = np.interp(right_lift, (0, 200), (0, 100))

            # Use the maximum percentage for progress and visualization
            per = max(left_per, right_per)
            bar = np.interp(per, (0, 100), (650, 100))

            # Determine color based on posture accuracy
            if per >= 100:  # Hand raised above shoulder
                color = (0, 255, 0)  # Green
                movement_completed = True
            else:
                color = (0, 0, 255)  # Red
            
            # Draw posture tracking lines
            # Left side: shoulder to elbow, elbow to wrist
            cv2.line(img, (lmList[11][1], lmList[11][2]), (lmList[13][1], lmList[13][2]), color, 5)
            cv2.line(img, (lmList[13][1], lmList[13][2]), (lmList[15][1], lmList[15][2]), color, 5)

            # Right side: shoulder to elbow, elbow to wrist
            cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[14][1], lmList[14][2]), color, 5)
            cv2.line(img, (lmList[14][1], lmList[14][2]), (lmList[16][1], lmList[16][2]), color, 5)

            # Logic for counting repetitions
            if movement_completed and per == 0:  # Reset position
                count += 1
                movement_completed = False

            # Progress bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Count display
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Show the updated video feed
        cv2.imshow("Hand Uplift Exercise", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def track_knee_raises():
    """Tracking logic for standing knee raises exercise with visual posture feedback."""
    cap = cv2.VideoCapture(0)
    detector = pm.poseDetector()
    count, pTime = 0, 0
    movement_completed = False

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to access the camera.")
            break

        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            # Track knee lift
            # Use landmarks:
            # 23 (left hip), 25 (left knee), 27 (left ankle)
            # 24 (right hip), 26 (right knee), 28 (right ankle)
            
            # Calculate vertical lift of knees relative to hips
            left_hip_y = lmList[23][2]
            right_hip_y = lmList[24][2]
            left_knee_y = lmList[25][2]
            right_knee_y = lmList[26][2]

            # Calculate knee lift percentage
            left_lift = max(0, left_hip_y - left_knee_y)
            right_lift = max(0, right_hip_y - right_knee_y)

            left_per = np.interp(left_lift, (0, 100), (0, 100))
            right_per = np.interp(right_lift, (0, 100), (0, 100))
            per = max(left_per, right_per)

            # Interpolate for progress bar
            bar = np.interp(per, (0, 100), (650, 100))

            # Set line color based on posture accuracy
            if per >= 70:  # Good posture threshold
                color = (0, 255, 0)  # Green
                movement_completed = True
            else:
                color = (0, 0, 255)  # Red
            
            # Draw posture tracking lines
            cv2.line(img, (lmList[23][1], lmList[23][2]), (lmList[25][1], lmList[25][2]), color, 5)  # Left hip to knee
            cv2.line(img, (lmList[25][1], lmList[25][2]), (lmList[27][1], lmList[27][2]), color, 5)  # Left knee to ankle
            cv2.line(img, (lmList[24][1], lmList[24][2]), (lmList[26][1], lmList[26][2]), color, 5)  # Right hip to knee
            cv2.line(img, (lmList[26][1], lmList[26][2]), (lmList[28][1], lmList[28][2]), color, 5)  # Right knee to ankle

            # Logic for counting reps
            if movement_completed and per <= 10:  # Reset threshold
                count += 1
                movement_completed = False

            # Progress bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Count display
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Show the updated video feed
        cv2.imshow("Knee Raises Exercise", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)
