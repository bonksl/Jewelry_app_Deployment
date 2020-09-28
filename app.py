from flask import Flask, render_template, request, flash, redirect, send_file
from werkzeug.utils import secure_filename
from data import Articles
import os
import urllib.request


from utils import *
import sys
#

import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback
from fastai.vision import load_learner 

import warnings
warnings.simplefilter("ignore")

import boto3
s3 = boto3.client('s3')

if not os.path.exists('./Jewelry_Recognition/jewelry/models/final_model.pth'):
	s3.download_file('bonaventure', 'final_model.pth', './Jewelry_Recognition/jewelry/models/final_model.pth')
if not os.path.exists('./models/final_model.pth'):
	s3.download_file('bonaventure', 'final_model.pth', './models/final_model.pth')


# Init app
app = Flask(__name__)
UPLOAD_FOLDER = "static/img/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


Articles = Articles()

#Upload API
@app.route('/Upload', methods=['GET','POST'])
def upload_file():
	if request.method == 'POST':
		#check if the post request has the file part
		if 'file' not in request.files:
			print ('no file')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also submit an empty part without file name
		if file.filename == '':
			print ('no filename')
			return redirect (request.url)
		else:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			print("saved file successfully")
			# send file name as parameter to download
			return redirect('/downloadfile/' + filename)
	return render_template('Upload.html')

#Download API
@app.route("/downloadfile/<filename>", methods = ['GET'])
def predict_img(filename):
	print("s1")

	path = 'Jewelry_Recognition/jewelry'

	print("s2")
	tfms = get_transforms(max_rotate= 10.,max_zoom=1., max_lighting=0.20, do_flip=False,
                      max_warp=0., xtra_tfms=[flip_lr(), brightness(change=(0.3, 0.60), p=0.7), contrast(scale=(0.5, 2), p=0.7),
                                              crop_pad(size=600, padding_mode='border', row_pct=0.,col_pct=0.),
                                              rand_zoom(scale=(1.,1.5)), rand_crop(),
                                              perspective_warp(magnitude=(-0.1,0.1)),
                                              symmetric_warp(magnitude=(-0.1,0.1)) ])
	
	print("s3")
	src = (ImageList.from_folder(path)
        .split_by_rand_pct(0.2, seed=42)
        .label_from_folder())
	print("s4")

	data = (src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128)
        .databunch(bs=64, num_workers=0)
        .normalize(imagenet_stats))
	print("s5")

	learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate])

	print("s6")
	learn.load('final_model')
	#import pdb; pdb.set_trace()
	print("s7")
	img = open_image(app.config['UPLOAD_FOLDER']+ '/' + filename)

	print("s8")
	predicted_class, predicted_index, outputs = learn.predict(img)

	print("s9")

	return render_template('Download.html', value = filename, predicted = data.classes[int(predicted_index)])


@app.route('/return-files/<filename>')
def return_files_tut(filename):
	file_path = UPLOAD_FOLDER + filename
	return send_file(file_path, as_attachment=True, attachment_filename='')


@app.route('/', methods =['GET'])
def index():
	return render_template('home.html')


@app.route('/home', methods =['GET'])
def home():
	return render_template('home.html')


@app.route('/About')
def about():
	return render_template('about.html')


@app.route('/Articles')
def articles():
	return render_template('articles.html', articles = Articles)


@app.route('/article/<string:id>/')
def article(id):
	return render_template('article.html', id=id)












# Run Server
if __name__ == '__main__':
	#app.run(host='0.0.0.0', port=33)
	app.run(debug=True, port=os.getenv("PORT", 5000))