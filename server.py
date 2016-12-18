import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug import secure_filename
import predict

UPLOAD_FOLDER = './uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = './static'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def upload_file():
    res = {
        'succ': False
    }
    print(request)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        res['succ'] = True
        res['clazz'] = str(predict.classify(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    return jsonify(res)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
