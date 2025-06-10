from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
import numpy as np
import os
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/pneumoniaApp"
mongo = PyMongo(app)

# Access users collection
users = mongo.db.users

# Load the trained model
model = load_model(r"C:\Users\HP\DL Project - 4\Pneumonia-Prediction\model weights\vgg_unfrozen.h5")
class_labels = {0: 'NORMAL', 1: 'PNEUMONIA'}

UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if users.find_one({'email': email}):
            return render_template('register.html', message="Email already exists.")

        if password != confirm_password:
            return render_template('register.html', message="Passwords do not match.")

        hashed_password = generate_password_hash(password)
        users.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password
        })

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = users.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']
            return redirect(url_for('upload_image'))
        else:
            message = 'Invalid email or password'
    return render_template('login.html', message=message)

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                img = image.load_img(file_path, target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                predicted_class = int(np.argmax(prediction))
                label = class_labels[predicted_class]
                confidence = round(np.max(prediction) * 100, 2)

                insert_result = mongo.db.predictions.insert_one({
                    "email": session['email'],
                    "filename": filename,
                    "label": label,
                    "confidence": float(confidence),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                return render_template('result.html', label=label, confidence=confidence, filename=filename, email=session['email'])

            except Exception as e:
                print("Error during prediction or saving to DB:", e)
                return render_template('upload.html', message="Error processing the image or saving prediction.")
        else:
            return render_template('upload.html', message="Please upload an image file.")

    return render_template('upload.html')

@app.route('/download_pdf/<filename>/<label>/<confidence>')
def download_pdf(filename, label, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True)
    pdf.cell(200, 10, txt=f"User: {session.get('email', 'Unknown')}", ln=True)
    pdf.cell(200, 10, txt=f"Result: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence}%", ln=True)
    pdf.image(f"static/uploads/{filename}", x=10, y=60, w=100)

    pdf_filename = f"report_{filename.split('.')[0]}.pdf"
    pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
    pdf.output(pdf_path)

    return redirect(url_for('static', filename=f'reports/{pdf_filename}'))

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    records = list(mongo.db.predictions.find({'email': session['email']}))
    return render_template('dashboard.html', records=records)

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
