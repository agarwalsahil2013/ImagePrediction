import os
#import cv2
from flask import Flask, request, render_template, redirect, url_for
#from main_model import predict_image
from imageai.Classification import ImageClassification

#execution_path = os.getcwd()
#b = 'mobilenet_v2.h5'
b = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'

path_model = '../master/'

prediction = ImageClassification()
#prediction.setModelTypeAsMobileNetV2()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(path_model,b))
prediction.loadModel()

#UPLOAD_FOLDER = './static/'

UPLOAD_FOLDER = 'static/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST','GET'])
def get_file():
    if request.method == 'POST':
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        return redirect(url_for('success', user=file1.filename))
    return render_template('index.html')

@app.route('/success/<user>')
def success(user):
	d = 'static/' + user
	e = prediction.classifyImage(d, result_count=1)
	result = f"Image is: {e[0][0].upper()} with {round(e[1][0],2)}% probability"
	return render_template('result.html', name=result, unn=user)

if __name__ == '__main__':
    app.run(debug=True)
