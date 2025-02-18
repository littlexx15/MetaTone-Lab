import os
import cv2
import dlib
import torch
import numpy as np
import gradio as gr
from torchvision import models, transforms
import ollama

# -------------------------------
# 1️⃣ 面部检测 & 情绪识别
# -------------------------------
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"❌ 找不到 {PREDICTOR_PATH}，请确保文件已下载！")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

emotion_model = models.resnet18(pretrained=True)
emotion_model.fc = torch.nn.Linear(512, 7)
emotion_model.eval()

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def detect_emotion(image_path):
    """使用 PyTorch 进行情绪识别"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = emotion_model(image)
        _, predicted = torch.max(outputs, 1)

    return emotion_labels[predicted.item()]

# -------------------------------
# 2️⃣ 面部特征提取
# -------------------------------
def extract_facial_features(image_path):
    """使用 OpenCV 和 dlib 提取面部特征"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    features = {"face_shape": "unknown", "skin_color": "unknown"}

    for face in faces:
        landmarks = predictor(gray, face)
        features["face_shape"] = "oval" if landmarks.part(0).x < landmarks.part(16).x else "round"
        features["skin_color"] = "light" if np.mean(gray) > 127 else "dark"
    
    return features

# -------------------------------
# 3️⃣ 生成歌词 (Ollama / Gemma:2b)
# -------------------------------
def generate_lyrics(facial_features, emotion):
    """结合面部特征和情绪生成歌词"""
    prompt = f"A poetic song about a person with {facial_features['face_shape']} face and {facial_features['skin_color']} skin, feeling {emotion}."
    
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    
    lyrics = response['message']['content']
    if len(lyrics.split()) < 15:
        lyrics += " This song is full of emotions and melodies that flow smoothly."
    
    return lyrics

# -------------------------------
# 4️⃣ Gradio 界面
# -------------------------------
def process_image(image):
    """完整的 AI 歌词生成流程"""
    cv2.imwrite("input.jpg", image)
      
    # 检测情绪
    emotion = detect_emotion("input.jpg")
    print(f"🤔 识别的情绪：{emotion}")  # ✅ 打印情绪结果

    # 提取面部特征
    features = extract_facial_features("input.jpg")
    print(f"📌 提取的面部特征：{features}")  # ✅ 打印面部特征
    
    # 生成歌词
    lyrics = generate_lyrics(features, emotion)

    return f"🎭 识别的情绪：{emotion}\n🖼 提取的面部特征：{features}\n🎶 生成的歌词：\n{lyrics}"

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="AI 歌词生成器",
    description="上传一张照片，AI 将根据你的面部特征生成一首歌词 🎵"
)

if __name__ == "__main__":
    print("🚀 Python 运行成功！")
    interface.launch()