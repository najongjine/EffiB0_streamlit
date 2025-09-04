import json
import numpy as np
import h5py
from PIL import Image
import streamlit as st
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers

IMG_H, IMG_W = 224, 224
MODEL_FILE = "EfficientNetB0.h5"
LABEL_FILE = "EfficientNetB0.json"

st.set_page_config(page_title="EfficientNetB0 이미지 분류기", page_icon="🧠", layout="centered")
st.title("🧠 EfficientNetB0 이미지 분류기 (호환 로더)")
st.caption("모델/라벨 버전 차이·헤드 구조 차이를 최대한 흡수하도록 설계한 로더입니다.")

def _load_labels(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        try:
            labels = [labels[str(i)] for i in range(len(labels))]
        except Exception:
            labels = list(labels.values())
    return labels

def _inspect_input_channels_from_h5(h5_path: Path) -> int | None:
    try:
        with h5py.File(str(h5_path), "r") as f:
            # keras saver는 가중치 경로가 구현/버전에 따라 달라질 수 있으므로 탐색 시도
            for k in f.keys():
                # stem_conv/kernel 이름 후보들을 스캔
                def walk(g):
                    for name, item in g.items():
                        if isinstance(item, h5py.Dataset) and name.endswith("kernel:0"):
                            # 커널 shape = (kh, kw, Cin, Cout)
                            shp = item.shape
                            if len(shp) == 4 and shp[-1] == 32 and shp[0] == 3 and shp[1] == 3:
                                return shp[2]  # Cin
                        elif isinstance(item, h5py.Group):
                            cin = walk(item)
                            if cin is not None:
                                return cin
                cin = walk(f[k]) if isinstance(f[k], h5py.Group) else None
                if cin is not None:
                    return int(cin)
    except Exception:
        pass
    return None

def _build_backbone(channels: int):
    # 학습 시 include_top=False 였을 가능성 높음 → backbone만 구성
    backbone = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_H, IMG_W, channels)
    )
    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    # 분류 헤드는 나중에 클래스 수를 알고 붙임
    return backbone, x

def _attach_head(x, num_classes: int):
    out = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    return out

@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    mpath = Path(MODEL_FILE)
    lpath = Path(LABEL_FILE)

    if not lpath.exists():
        raise FileNotFoundError(f"라벨 파일 없음: {lpath.resolve()}")
    class_names = _load_labels(lpath)
    num_classes = len(class_names)
    if num_classes < 2:
        st.warning("라벨 수가 2 미만입니다. 라벨 JSON을 확인하세요.")

    if not mpath.exists():
        raise FileNotFoundError(f"모델 파일 없음: {mpath.resolve()}")

    # A) 모델 전체 로딩 (학습 당시 SavedModel/h5 full model인 경우)
    try:
        model = load_model(str(mpath), compile=False)
        return model, class_names
    except Exception as eA:
        st.info(f"직접 load_model 실패 → 백본 재구성 후 by_name 로딩 시도. 에러: {eA}")

    # B) H5에서 stem_conv 커널로 입력 채널 추정
    in_ch = _inspect_input_channels_from_h5(mpath) or 3
    st.write(f"추정 입력 채널: {in_ch}")

    # C) include_top=False 백본 만들고, by_name=True + skip_mismatch=True로 최대한 주입
    backbone, feat = _build_backbone(in_ch)
    try:
        backbone.load_weights(str(mpath), by_name=True, skip_mismatch=True)
        st.success("백본 가중치(by_name, skip_mismatch) 일부/전체 주입 성공.")
    except Exception as eB:
        st.warning(f"백본 가중치 주입 실패(계속 진행): {eB}")

    # D) 분류 헤드 부착
    logits = _attach_head(feat, num_classes)
    model = Model(backbone.input, logits, name="effib0_recovered")

    # (선택) 추론용 compile
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    return model, class_names

# ====== UI ======
try:
    model, class_names = load_model_and_labels()
    st.success("✅ 모델/라벨 로딩 완료!")
except Exception as e:
    st.error(f"모델/라벨 로딩 실패: {e}")
    st.stop()

uploaded = st.file_uploader("이미지 업로드 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    in_channels = model.input_shape[-1]
    img = Image.open(uploaded)
    img = img.convert("L" if in_channels == 1 else "RGB")

    st.image(img, caption="업로드 이미지", use_container_width=True)

    img_resized = img.resize((IMG_W, IMG_H))
    img_arr = kimage.img_to_array(img_resized)
    img_arr = preprocess_input(img_arr)
    img_batch = np.expand_dims(img_arr, axis=0)

    with st.spinner("예측 중..."):
        probs = model.predict(img_batch, verbose=0)[0]

    top = int(np.argmax(probs))
    pred = class_names[top] if top < len(class_names) else f"Class {top}"

    st.subheader("🔮 예측 결과")
    st.write(f"**Predicted:** {pred}")

    try:
        import pandas as pd
        k = min(len(probs), len(class_names))
        df = pd.DataFrame({"class": class_names[:k], "probability": probs[:k]}).sort_values("probability", ascending=False)
        st.dataframe(df.reset_index(drop=True), use_container_width=True)
        st.bar_chart(df.set_index("class"))
    except Exception:
        st.write("클래스별 확률:")
        for i in np.argsort(-probs[:len(class_names)]):
            st.write(f"- {class_names[i]}: {probs[i]:.4f}")
else:
    st.info("오른쪽에서 이미지를 업로드하세요. 모델/라벨은 이 파일과 같은 폴더에 두세요.")
