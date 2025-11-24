import cv2
import av
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Define the callback function for video processing
def video_frame_callback(frame):
    # Convert frame to numpy array (OpenCV format)
    img = frame.to_ndarray(format="bgr24")
    
    # Flip the image horizontally for a selfie-view display
    img = cv2.flip(img, 1)
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(img_rgb)
    
    # Draw Hand Landmarks & Logic
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton on the hand
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # --- YOUR PREDICTION LOGIC GOES HERE ---
            # Example: extracting coordinates for your model
            # landmarks = []
            # for lm in hand_landmarks.landmark:
            #     landmarks.append([lm.x, lm.y, lm.z])
            # prediction = model.predict([landmarks]) 
            
            # Placeholder text (Replace with your actual prediction)
            cv2.putText(img, "Hand Detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Return the processed frame back to the browser
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Streamlit Layout
st.title("Real-Time Hand Sign Detection")
st.write("Turn on the webcam to detect hand signs in real-time.")

# 4. WebRTC Configuration (STUN server for cloud deployment reliability)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 5. Render the Webcam Stream
webrtc_streamer(
    key="hand-sign-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)