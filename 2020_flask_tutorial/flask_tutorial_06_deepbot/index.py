#!/usr/bin/python
# -*- coding: utf-8 -*
'''
api.py: DeepBot Api Server
'''

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_url_path='')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
      return render_template('register.html')


@app.route('/deepbot', methods=['GET', 'POST'])
def deepbot():
    user = request.form['user']
    # pwd = request.form['p']
    return render_template('deepbot.html',user=user)
     

@app.route('/message', methods=['GET', 'POST'])
def chatting():
    # request
    message = request.json['message']
    return jsonify({'message' : message})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
