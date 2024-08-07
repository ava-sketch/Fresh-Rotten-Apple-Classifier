import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YZa2PiC7CmoBlhhkkNca"
)

# Function to run inference on an image
def run_inference(image):
    temp_image_file = "temp_image.webp"
    cv2.imwrite(temp_image_file, image)
    results = CLIENT.infer(temp_image_file, model_id="rotten-apple-detection-ve9cr/3")
    return results

# Function to annotate image with bounding boxes and labels
def annotate_image(image, results):
    if 'predictions' in results:
        detections = sv.Detections.from_inference(results)
        
        # Create supervision annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Annotate the image with inference results
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        return annotated_image
    else:
        return image

# Streamlit app
st.title("Rotten/Fresh apple identification interface")

# Option to choose between file upload and webcam
mode = st.radio("Choose input method:", ("Upload an Image", "Use Webcam"))

if mode == "Upload an Image":
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        # Read image with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display uploaded image
        st.image(image, channels="BGR", caption='Uploaded Image.', use_column_width=True)

        # Run inference on the image
        with st.spinner('Processing...'):
            results = run_inference(image)

        # Display results
        if 'predictions' in results:
            class_labels = [prediction['class'] for prediction in results['predictions']]
            st.write("Class Labels:", class_labels)
            
            # Annotate the image with inference results
            annotated_image = annotate_image(image, results)

            # Display the annotated image
            st.image(annotated_image, channels="BGR", caption='Annotated Image.', use_column_width=True)
        else:
            st.write("No predictions found in the result.")

elif mode == "Use Webcam":
    # Webcam streaming setup
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")

            # Run inference on the webcam frame
            results = run_inference(image)

            # Annotate the image with inference results if there are any predictions
            annotated_image = annotate_image(image, results)
            return annotated_image

    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Stream webcam video
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.result_queue = webrtc_ctx.result_queue
