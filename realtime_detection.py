import cv2
import numpy as np
import mediapipe as mp
import pickle


# Load the Random Forest model and labels (to be REPLACED with deep learning model)
labels = {
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h", "i": "i", 
    "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "q", "r": "r", 
    "s": "s", "t": "t", "u": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z",
    "1": "BackSpace", "2": "Clear", "3": "Space", "4": "" # "4" represents no hand detected.
}

with open("./asl_prediction/ASL_model.p", "rb") as f: # Model path
    model = pickle.load(f)
rf_model = model["model"] # Adapt this for deep learning model


# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)


# Global variables (Shared with application.py)
predicted_text = ""
same_characters = ""
final_characters = ""
count = 0


def update_frame_for_web(hand_landmarks, rf_model):  # function for web app
    global predicted_text, same_characters, final_characters, count
    x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
    y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
    min_x, min_y = min(x_coordinates), min(y_coordinates)

    normalized_landmarks = []
    for coordinates in hand_landmarks.landmark:
        normalized_landmarks.extend([
            coordinates.x - min_x,
            coordinates.y - min_y
        ])

    sample = np.asarray(normalized_landmarks).reshape(1, -1)
    predicted_character = rf_model.predict(sample)[0] #Change as per model

    if predicted_character != "4":
        predicted_text += predicted_character

        if predicted_text[-1] != predicted_text[-2]:
            count = 0
            same_characters = ""
        else:
            same_characters += predicted_character
            count += 1

        if count == 30:
            if predicted_character == "1":
                if final_characters:
                    final_characters = list(final_characters)
                    final_characters.pop()
                    final_characters = "".join(final_characters)

            elif predicted_character == "2":
                final_characters = ""

            elif predicted_character == "3":
                final_characters += " "

            else:
                final_characters += str(list(set(same_characters))[0])


            count = 0
            same_characters = ""
            predicted_text = ""  # Reset predicted_text
    return predicted_character



def release_video():
    cap.release()  #Release if had initialized video capture in this file, otherwise this is not used
    cv2.destroyAllWindows()
