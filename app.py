# pip install streamlit tensorflow pillow
# streamlit run vgg16_predict.py

import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

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
    ["bearded_fireworm.png", "blue_dragon_sea_slug.png",
     "matamata_turtle.png", "pink_fairy_armadilo.png"],
    caption=["bearded_fireworm", "blue_dragon_sea_slug",
             "matamata_turtle", "pink_fairy_armadilo"],
    width=300
)

MODEL_PATH = "EffiB0_test.h5"
LABEL_PATH = "EffiB0_test.json"

def normalize_class_names(raw):
    # raw가 리스트면 그대로, dict면 인덱스 오름차순으로 정렬해 리스트화
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # 키가 문자열일 수도 있으므로 int 변환 시도 -> 실패 시 문자열 정렬
        try:
            items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
            return [v for _, v in items]
        except Exception:
            return [raw[k] for k in sorted(raw.keys())]
    raise ValueError("EffiB0_test.json 형식이 list 또는 dict가 아닙니다.")

def build_base_model(num_classes: int) -> tf.keras.Model:
    # 학습 당시와 최대한 유사하게 구성 (채널=3, 224x224)
    # include_top=True로 num_classes에 맞는 Dense(softmax)까지 생성
    return EfficientNetB0(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=(224, 224, 3)
    )

@st.cache_resource
def load_model_and_labels():
    # 1) 라벨 로드
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    class_names = normalize_class_names(raw)
    num_classes = len(class_names)

    # 2) 모델 로드 (3단계 전략)
    info_msgs = []

    # 2-1) 전체 모델 복원 (가장 깔끔)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        info_msgs.append("load_model() 성공: 저장된 전체 모델 구조를 복원했습니다.")
        # 최종 Dense 크기와 클래스 수가 다른 경우 대비(드물지만)
        if model.output_shape[-1] != num_classes:
            info_msgs.append(
                f"경고: 모델의 출력 크기({model.output_shape[-1]})와 라벨 수({num_classes})가 다릅니다."
            )
        return model, class_names, info_msgs
    except Exception as e:
        info_msgs.append(f"load_model() 실패: {e}")

    # 2-2) 동일 구조로 빌드 후 가중치 주입
    try:
        model = build_base_model(num_classes)
        model.load_weights(MODEL_PATH)
        info_msgs.append("동일 구조 빌드 + load_weights() 성공.")
        return model, class_names, info_msgs
    except Exception as e:
        info_msgs.append(f"load_weights() (정확 일치) 실패: {e}")

    # 2-3) 이름 기반 부분 로드(불일치 무시) — 베이스는 들어오고, 헤드는 현 코드 유지
    try:
        model = build_base_model(num_classes)
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        info_msgs.append("이름 기반 부분 로드(by_name=True, skip_mismatch=True) 성공.")
        info_msgs.append("일부 레이어는 매칭되지 않아 현재 헤드 구조를 사용합니다.")
        return model, class_names, info_msgs
    except Exception as e:
        info_msgs.append(f"부분 로드도 실패: {e}")
        raise RuntimeError("\n".join(info_msgs))

model, class_names, load_infos = load_model_and_labels()
for m in load_infos:
    st.info(m)

# 사용자 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 로딩
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='업로드된 이미지', use_column_width=True)

    # 전처리
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)                # (224, 224, 3)
    img_array = preprocess_input(img_array)            # EfficientNet 전처리
    img_array = np.expand_dims(img_array, axis=0)      # (1, 224, 224, 3)

    # 예측
    predictions = model.predict(img_array)
    probs = predictions[0]
    predicted_idx = int(np.argmax(probs))
    predicted_class = class_names[predicted_idx]
    max_confidence = float(np.max(probs))

    st.write(f"predictions : {predictions}")
    if max_confidence < 0.6:
        st.markdown("## 😥 학습한 클래스가 아니거나, 분류를 실패했습니다")
    else:
        st.markdown(f"### ✅ 예측 결과: **{predicted_class}** ({max_confidence:.4f})")
        st.markdown("### 🔢 클래스별 확률")
        for i, prob in enumerate(probs):
            st.write(f"{class_names[i]}: {prob:.4f}")
