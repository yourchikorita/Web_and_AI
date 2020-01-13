from flask import Flask, render_template, request
from werkzeug import secure_filename
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
# from IPython.display import display, Image
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['GET','POST'])
def upload_file():
    if request.method =='POST':
        new_model = keras.models.load_model('rsp.h5')
        # predicting images
       


        #브라우저에서 업로드 한 사진 저장하는 디렉토리 
        f=request.files['file']
        file_dir='./picture/' 
        f.save(file_dir+secure_filename(f.filename))

        print('방금 올린파일 filename==',f.filename)

        #방금 올린 파일 모델에 넣어서 예측해보기
        now_i_upload=f.filename
        path = "./picture/"+now_i_upload
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = new_model.predict(images, batch_size=10)
        # paper.rock.scissor
        test='angel'
        return render_template('result.html',file_dir=file_dir,classes=classes,test=test)

if __name__ == '__main__':
    app.run()