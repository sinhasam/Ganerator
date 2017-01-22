import subprocess

from flask import Flask, jsonify, render_template

# Stuff for contact form
#import os
#from flask import request, redirect, url_for, send_from_directory
#from werkzeug.utils import secure_filename

app = Flask(__name__)

#UPLOAD_FOLDER = '/path/to/the/uploads'
#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

#app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#def allowed_file(filename):
#	return '.' in filename and \
#		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# more code for request goes here

#@app.route('/uploads/<filename>')
#def uploaded_file(filename):
#	return send_from_directory(app.config['UPLOAD_FOLDER'],
#		filename)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/generate')
def generate():
	subprocess.call(["lua", "noise.lua"])
	return jsonify(result='generated image from noise')

@app.route('/upload')
def upload():
	subprocess.call(["lua", "upload.lua"])
	return jsonify(result='uploaded image')

if __name__ == "__main__":
	app.run()
