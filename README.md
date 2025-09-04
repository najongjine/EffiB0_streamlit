pip install uv

uv init

uv python install 3.10 
uv python pin 3.10 
uv venv --python 3.10

uv add -r requirements.txt


Keras 로 파인튜닝한 모델이 colab 이외의 환경에선 각종 모듈 버전문제가 심해서 pytorch 파인튜닝 모델로 바꿨습니다.