#!/usr/bin/python
# -*- coding: utf-8 -*
'''
api.py: DeepBot Api Server
'''

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    myList=[]
    for i in range(0,10):
        myList.append('count :  %d' % i )

    myDict ={"a" : "apple","b":"banana"}    



    return render_template('login.html',myList=myList,myDict=myDict)



# 실습. 앞시간에 만들었던 login페이지와 flask 를 활용한 프로젝트하기

if __name__ == '__main__':
    app.run(debug=True, port=5000)
