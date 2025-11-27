import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import av
import tensorflow as tensorflow # Change if using PyTorch/other

# ------------------------------------------------
# 1. LOAD YOUR MODEL (Cached for performance)
# ------------------------------------------------
@st.cache_resource
def load_model():
    # REPLACE WITH YOUR MODEL LOADING LOGIC
    model = tensorflow.keras.models.load_model('model.h5') 
    return model

model = load_model()

# ------------------------------------------------
# 2. DEFINE THE VIDEO PROCESSOR
# ------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to a numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        
        # ------------------------------------------------
        # START: YOUR DETECTION LOGIC
        # ------------------------------------------------
        
        # 1. Preprocess (Resize/Normalize based on your model training)
        # Example:
        img_input = cv2.resize(img, (224, 224))
        img_input = np.expand_dims(img_input, axis=0) / 255.0
        
        # 2. Predict
        prediction = model.predict(img_input)
        class_index = np.argmax(prediction)
        
        # 3. Map index to label (Update your labels here)
        labels = ["Hello", "Yes", "No", "Thanks", "I Love You"] 
        detected_text = labels[class_index]
        confidence = float(np.max(prediction))

        # ------------------------------------------------
        # END: YOUR DETECTION LOGIC
        # ------------------------------------------------

        # Draw the result on the original frame
        if confidence > 0.7: # Only show if confident
            cv2.putText(img, f"Sign: {detected_text} ({confidence:.2f})", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Optional: Draw rectangle if you have bounding box logic
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        # Return the processed frame to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------------------------------------
# 3. BUILD THE WEBPAGE
# ------------------------------------------------
st.title("🤟 Live Hand Sign Detection")
st.write("Turn on your webcam to detect hand signs in real-time.")

# This block handles the WebRTC streaming
webrtc_streamer(
    key="hand-sign-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration={  # Add this config for cloud deployment reliability
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.sidebar.title("Model Info")
st.sidebar.info("This model detects basic ASL signs. Ensure your hand is clearly visible.")