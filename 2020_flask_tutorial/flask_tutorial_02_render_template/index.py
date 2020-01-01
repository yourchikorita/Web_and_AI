#!/usr/bin/python
# -*- coding: utf-8 -*

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)


# 실습1. login.html파일을 style.css 파일과 연결해서 body 의 background-color 변경하기
