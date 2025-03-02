from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename  # <-- Add this import
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part", 400
        
        file = request.files['file']
        
        # If no file is selected, return an error
        if file.filename == '':
            return "No selected file", 400
    
        filename = secure_filename(file.filename)

        # Create the upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the file into the static/uploads folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        return "done"

@app.route('/')
def pints_sig():
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
