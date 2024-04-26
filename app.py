from flask import Flask, render_template, request
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='/templates')
app = Flask(__name__, static_url_path='/static') 

@app.route('/')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('files')  # Get a list of uploaded files
        if files:
            for file in files:
                # Process each uploaded file (e.g., save, validate)
                # ... your logic here ...
                file.save(secure_filename(file.filename))  # Example: saving files
            flash("Files uploaded successfully!", "success")
            return redirect(url_for('home'))  # Redirect back to home
        else:
            flash("No files selected!", "danger")
            return redirect(url_for('home'))  # Redirect back to home
    else:
        return "Invalid request method"

@app.route("/")
def home():
    return render_template('home.html')

if __name__ == "__main__":
  app.run(debug=True)  # Run the app in debug mode
