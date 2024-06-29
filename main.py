import cv2
import os
from deepface import DeepFace

# Define function to detect faces and classify emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_and_classify_emotion(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)

    # Detect faces and classify emotions
    results = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

    # Process each face detected
    for result in results:
        box = result['region']
        emotion_label = result['dominant_emotion']
        (x, y, w, h) = box['x'], box['y'], box['w'], box['h']

        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save output image
    cv2.imwrite(output_path, image)
    print(f"Output saved to: {output_path}")

# Input and output paths
input_image_path = 'faces/person7.jpg'  # Change to your input image path
output_image_path = 'output/person7_output.jpg'  # Change to your desired output path

# Call the function
detect_and_classify_emotion(input_image_path, output_image_path)
