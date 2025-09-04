import torch, json
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

# ✅ 경로 설정 (로컬 기준)
CKPT_PATH = "efficientnet_b0_wierd_animals.pt"
CLASS_PATH = "efficientnet_b0_wierd_animals.json"
IMG_PATH   = "C:/Users/itg/Pictures/Blue_Dragon_Sea_Slug1.png"  # ← 같은 폴더에 있는 이미지

# ✅ 전처리 정의 (학습 때와 동일하게)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ✅ 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.to(device).eval()

# ✅ 이미지 예측
img = Image.open(IMG_PATH)
x = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

with torch.no_grad():
    prob = F.softmax(model(x), dim=1)[0]
    idx = int(prob.argmax().item())
    print(f"✅ 예측: {class_names[idx]} ({prob[idx]*100:.1f}%)")


