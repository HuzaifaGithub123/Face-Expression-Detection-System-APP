from flask import Flask, request, render_template, url_for
from keras.api.preprocessing import image
from keras.api.preprocessing.image import load_img, img_to_array
from keras.api.models import load_model
import numpy as np
import os
from PIL import Image

# Flask App
app = Flask(__name__)

#model
model = load_model('.venv\\facial_emotion_detection_model.h5')

#Define class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index =np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    return predicted_class, confidence


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No File Uploaded'
        file = request.files['file']
        if file.filename == '':
            return 'No File Selected'
        # Process the file since checks passed
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        emotion, confidence = detect_emotion(file_path)
        return render_template('index.html', image_path=file_path, confidence=confidence, emotion=emotion)
    # Handle GET requests
    return render_template('index.html')
#Python main call
if __name__ == '__main__':
    app.run(debug=True)