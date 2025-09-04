# app.py
import io
import json
from typing import List, Tuple
import pandas as pd

import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from torchvision import models, transforms


st.set_page_config(page_title="Weird Animals EfficientNetB0", layout="centered")

# ====== 고정 경로 (소스와 같은 폴더) ======
CKPT_PATH = "efficientnet_b0_wierd_animals.pt"
CLASS_PATH = "efficientnet_b0_wierd_animals.json"

# ====== 전처리 (학습과 동일) ======
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ====== 유틸 ======
def strip_module_prefix(state_dict: dict) -> dict:
    """Distributed/DataParallel 로 저장된 ckpt의 'module.' prefix 제거."""
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

@st.cache_data(show_spinner=False)
def load_class_names(path: str) -> List[str]:
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

@st.cache_resource(show_spinner=True)
def load_model(num_classes: int, device: torch.device):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    return model

def run_inference(model: torch.nn.Module, img: Image.Image, device: torch.device) -> torch.Tensor:
    x = TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0].detach().cpu()
    return prob  # (C,)

def topk(prob: torch.Tensor, k: int = 5):
    vals, idxs = torch.topk(prob, k)
    return [(i.item(), v.item()) for v, i in zip(vals, idxs)]


# ====== UI ======
st.title("🐉 Weird Animals Classifier (EfficientNet-B0)")
st.caption("모델/클래스 파일은 app.py와 같은 폴더에 두고, 이미지만 업로드하면 예측합니다.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🖥️ Device: **{device.type.upper()}**")

# 클래스/모델 미리 로딩
try:
    class_names = load_class_names(CLASS_PATH)
    model = load_model(len(class_names), device)
except Exception as e:
    st.error(f"모델/클래스 로드 실패: {e}")
    st.stop()

uploaded_image = st.file_uploader("이미지 업로드 (png/jpg)", type=["png", "jpg", "jpeg"])

if not uploaded_image:
    st.info("⬆️ 예측할 이미지를 업로드하세요.")
    st.stop()

# 이미지 열기
try:
    img = Image.open(io.BytesIO(uploaded_image.read()))
except Exception as e:
    st.error(f"이미지 열기 실패: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("입력 이미지")
    st.image(img, use_container_width=True)

with col2:
    st.subheader("예측 결과")
    with st.spinner("모델 추론 중..."):
        prob = run_inference(model, img, device)

    # Top-1
    idx = int(prob.argmax().item())
    conf = float(prob[idx].item()) * 100
    st.markdown(f"**✅ 예측: {class_names[idx]}**  \n신뢰도: **{conf:.2f}%**")

    # Top-5 테이블 + 차트
    k = min(5, len(class_names))
    tk = topk(prob, k=k)
    labels = [class_names[i] for i, _ in tk]
    scores = [round(p * 100, 2) for _, p in tk]

    st.write("Top-k 확률:")
    df = pd.DataFrame({"Class": labels, "Confidence (%)": scores})

    # 표
    st.dataframe(df, hide_index=True, use_container_width=True)

    # ✅ 바 차트: 클래스명을 인덱스로 사용
    st.bar_chart(df.set_index("Class"))


st.caption("※ ckpt 키가 일부 달라도 strict=False로 로드합니다. CUDA 가능 시 자동으로 GPU 사용.")
