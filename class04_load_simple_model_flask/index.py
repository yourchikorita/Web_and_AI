from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World! Go to Login page!'

@app.route('/login')
def test():
    return render_template('login.html')


@app.route('/result', methods=['POST','GET'])
def post():


    new_model = keras.models.load_model('simplestmodel.h5')
    # new_model.summary()
   
    
    #POST방식
    # value = request.form['test']
    #GET방식
    value = request.args.get('test')
    print('value=',value)
    value_to_float= float(value)
    print(new_model.predict([value_to_float]))
    mypredic = new_model.predict([value_to_float])
    #login.html에서 메소드를 get으로 변경후 
    
    return render_template('result.html',value=value,mypredic=mypredic)

if __name__ == '__main__':
    app.run()