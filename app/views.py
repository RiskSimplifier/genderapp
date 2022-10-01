import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as mating


UPLOAD_FOLDER = 'static/upload'


def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        #save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path) # save image into upload folder
        # get prediction
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
      

        #generate report
        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image(array)
            eigen_image = obj['eig_img'].reshape(100,100)# eigen image(array)
            gender_name = obj['prediction_name'] # name
            score = round(obj['score']*100,2)# probability score

        # save grayscale and eigen in predict folder
        gray_image_name = f'roi_{1}.jpg'
        eig_image_name = f'eigen_{1}.jpg'
        mating.imsave(f'./static/predict/{gray_image_name}', gray_image, cmap='gray')
        mating.imsave(f'./static/predict/{eig_image_name}', eigen_image, cmap='gray')

        # save reoirt
        report.append([gray_image_name,
                       eig_image_name,
                       gender_name,
                        score])
                        
    return render_template('gender.html', fileupload=True, report=report) # POST Request

   