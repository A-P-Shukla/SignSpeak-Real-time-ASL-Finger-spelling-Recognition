from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import pickle
from realtime_detection import update_frame_for_web, release_video  # Import the new function


app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the Random Forest model and labels (to be REPLACED with deep learning model later)
labels = {
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h", "i": "i", 
    "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "q", "r": "r", 
    "s": "s", "t": "t", "u": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z",
    "1": "BackSpace", "2": "Clear", "3": "Space", "4": ""
}

with open("./asl_prediction/ASL_model.p", "rb") as f: # Model path
    model = pickle.load(f)
rf_model = model["model"]  # Adapt as needed for deep learning model

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables (These are now shared with realtime_detection.py)
predicted_text = ""
same_characters = ""
final_characters = ""
count = 0

CLASSES = list(labels.keys())  # Might not be need this with a deep learning model

# Flask Routes
@app.route('/')
def index():
    return render_template('Final_ASL/opening_page.html')  

@app.route('/opening_page.html')
def opening_page():
    return render_template('Final_ASL/opening_page.html')

@app.route('/signspeak.html')
def signspeak():
    return render_template('Final_ASL/signspeak.html')

@app.route('/contact_us.html')
def contact_us():
    return render_template('Final_ASL/contact_us.html')


@app.route('/predict/', methods=['POST'])
def predict():
    global final_characters #Access the global variable
    return jsonify({'predicted_string': final_characters}) #Send predicted string as json to front-end


@app.route('/clear', methods=['POST']) #Route to clear `final_characters` in backend
def clear_text():
    global final_characters
    final_characters = ""
    return jsonify({'message': 'Text area cleared'}) #Return success message


def gen_frames():
    global final_characters, predicted_text, same_characters, count #Access the global variables
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                try:
                    predicted_character = update_frame_for_web(hand_landmarks, rf_model) #Call the function to update string in realtime_detection.py

                    #Displaying predicted character on each frame
                    text_position = (20, 20)  # Top-left corner of the text background
                    background_color = (255, 255, 255)  # Background color (white)
                    text_color = (0, 0, 0)  # Text color (black)
                    font_scale = 1
                    thickness = 2

                    # Calculate the width and height of the text box
                    (text_width, text_height), baseline = cv2.getTextSize(predicted_character, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # Calculate bottom-right corner for the background rectangle based on text size
                    background_top_left = text_position
                    background_bottom_right = (text_position[0] + text_width + 180, text_position[1] + text_height + 10)

                    # Draw the filled rectangle as the background for text
                    cv2.rectangle(frame, background_top_left, background_bottom_right, background_color, -1)

                    # Draw the text on top of the rectangle
                    cv2.putText(
                        img=frame,
                        text=labels[predicted_character],
                        org=(text_position[0] + 5, text_position[1] + text_height),  # Adjust for padding within rectangle
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=text_color,
                        thickness=thickness,
                        lineType=cv2.LINE_AA
                    )


                except Exception as e:
                    print(f"Error during prediction in gen_frames: {e}")



        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True, threaded=True)