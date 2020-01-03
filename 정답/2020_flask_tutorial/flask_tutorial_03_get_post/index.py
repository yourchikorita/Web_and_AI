from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World! Go to Login page!'

@app.route('/login')
def test():
    return render_template('login.html')


@app.route('/result', methods=['POST','GET'])
def post():
    #POST방식
    # value = request.form['test']
    #GET방식
    value = request.args.get('test')
    #login.html에서 메소드를 get으로 변경후 
    
    return render_template('result.html',value=value)

if __name__ == '__main__':
    app.run()