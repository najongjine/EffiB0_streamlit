# vgg16_predict.py
# pip install streamlit tensorflow pillow huggingface_hub
# streamlit run vgg16_predict.py
# https://vgg16app-bgnstkinr6jtirpwqqmfdp.streamlit.app/

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import json
from PIL import Image
import io

st.title("Effi 이미지 분류기")
st.write("""
이 이미지 분류기는 
         Bearded_Fireworm, 
         Blue_Dragon_Sea Slug, 
         Mata_Mata_Turtle, 
         Pink_Fairy Armadillo 
분류기에요
""")
st.image(
    ["bearded_fireworm.png", "blue_dragon_sea_slug.png"
     ,"matamata_turtle.png","pink_fairy_armadilo.png"],
    caption=["bearded_fireworm", "blue_dragon_sea_slug"
             ,"matamata_turtle","pink_fairy_armadilo"],
    #use_column_width=True
    width=300
)

# 모델 및 클래스 불러오기
@st.cache_resource
def load_model_and_labels():
    model = load_model("EfficientNetB0_wierd_animals.h5")
    with open("EfficientNetB0_wierd_animals.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_labels()

# 사용자 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 로딩
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='업로드된 이미지', use_column_width=True)

    # 전처리
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"predictions : {predictions}") 
    #predictions : [[0.2519071 0.19521904 0.24748924 0.30538464]]
    max_confidence=predictions[0]
    max_confidence=max(max_confidence)
    #max_confidence : 0.30538463592529297
    if(max_confidence<0.6):
        st.markdown("## :( 학습한 클래스가 아니거나, 분류를 실패 했습니다")
    else:
        st.markdown(f"### ✅ 예측 결과: **{predicted_class}**")
        st.markdown("### 🔢 클래스별 확률")
        for i, prob in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {prob:.4f}")

    
