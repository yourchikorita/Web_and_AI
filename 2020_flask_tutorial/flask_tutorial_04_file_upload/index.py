from flask import Flask, render_template, request
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['GET','POST'])
def upload_file():
    if request.method =='POST':
        f=request.files['file']
        # file_dir='./' # 현재 위치에 파일 저장
        file_dir='C:/Users/EJ/Desktop/' #경로설정
        f.save(file_dir+secure_filename(f.filename))
        return render_template('result.html',file_dir=file_dir)

if __name__ == '__main__':
    app.run()