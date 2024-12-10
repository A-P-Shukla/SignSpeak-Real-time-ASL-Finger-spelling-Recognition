# SignSpeak: Real-time ASL Finger-spelling Recognition

SignSpeak is a web application designed to translate American Sign Language (ASL) finger-spelling into text in real-time, using a webcam as input. This project aims to bridge the communication gap between the deaf and hearing communities by providing a convenient and accessible tool for seamless interaction.  It leverages MediaPipe for robust hand tracking and a machine learning model for character prediction. The current implementation uses a Random Forest model as a placeholder, but the architecture is designed for easy integration with more sophisticated deep learning models for improved accuracy and performance.

## Features

* **Real-time Translation:**  Captures video from your webcam and translates finger-spelled letters into text instantly, enabling dynamic and engaging conversations.
* **Dynamic Text Area Display:** Presents the translated text in a user-friendly text area, which updates in real-time as you sign.
* **Integrated Hand Tracking:** Utilizes MediaPipe, a powerful library for building cross-platform, customizable ML solutions, to accurately track hand movements and extract relevant landmarks for prediction.
* **Flask Backend:**  A lightweight and flexible Flask backend handles video streaming, prediction requests from the frontend, and manages the application's routing.
* **Deep Learning Ready:** Designed with future scalability in mind. The architecture makes it simple to replace the placeholder Random Forest model with more advanced deep learning models (CNNs, RNNs, Transformers), allowing you to leverage the power of deep learning for enhanced accuracy.
* **"Clear" Functionality:** Includes a "clear" feature, currently handled by the prediction model itself.  This can be adapted to client-side control if your chosen model doesn't provide this functionality directly.


## Project Structure

The project is organized as follows:

* **`application.py`:**  The heart of the backend, built with Flask.  Handles:
    * Video streaming from the webcam.
    * Receiving prediction requests from the frontend.
    * Routing between different pages of the application.
    * Managing the core prediction loop and communication with `realtime_detection.py`.
* **`realtime_detection.py`:** Contains the core prediction logic:
    * Processes hand landmarks extracted by MediaPipe.
    * Prepares the data for the machine learning model.
    * Performs the prediction using the loaded model.
    * Manages the predicted text string, including handling the "clear" functionality.
* **`templates/Final_ASL/`:** Houses the HTML templates for the web pages:
    * `opening_page.html`:  Provides an overview of the project, its aims, and the team.
    * `signspeak.html`: The main page where the real-time translation takes place, displaying the video feed and the predicted text area.
    * `contact_us.html`:  Includes contact details for the project team.
* **`static/Final_ASL/`:**  Contains the static assets for the web application:
    * `signspeak.css`:  Stylesheets for visual presentation.
    * `signspeak.js`:  Frontend JavaScript code that handles:
        * Displaying the video stream.
        * Fetching predictions from the backend.
        * Updating the text area with the translated text.
    * Other static assets, such as images, icons, and the site manifest.
* **`asl_prediction/ASL_model.p`:**  The pickled machine learning model file.  The current implementation uses a Random Forest.  This is a *placeholder* and **must be replaced** with your trained deep learning model.  Adapt the loading and prediction parts of `application.py` and `realtime_detection.py` to use your specific model format.
## Installation

These steps will guide you through setting up and running the SignSpeak application.  We highly recommend using a virtual environment to manage project dependencies.

**1. Clone the repository:**
`https://github.com/A-P-Shukla/SignSpeak-Real-time-ASL-Finger-spelling-Recognition.git`

**2. Set up and activate a virtual environment:**
* `python3 -m venv .venv`
* `.venv/bin/activate  # Activates (Linux/macOS) `  
* `.venv\Scripts\activate # Activates (Windows)`      
  
**3. Install project dependencies:**
* `pip install -r requirements.txt`

## Contributing to the repo
* Contributions are welcome! Open an issue to discuss ideas or submit a pull request.
