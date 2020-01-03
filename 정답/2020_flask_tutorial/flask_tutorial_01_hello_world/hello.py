#import
from flask import Flask
#flask객체생성
app = Flask(__name__)

#route설정
# 데코레이터(Decorator)란?
# 하나의 함수를 취해서 또 다른 함수를 반환하는 함수를 의미합니다.
@app.route('/hoho')
#함수설정
def hello_world():
    return 'Hello world~~~~'



	
#서버 동작시키기
if __name__ == '__main__':
    app.run()

#실습1 다른 라우터 사용해보기