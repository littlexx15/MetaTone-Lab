import os
import cv2
import dlib
import torch
import numpy as np
import face_recognition
import torchaudio
from torchvision import models, transforms
import ollama
import gradio as gr

# -------------------------------
# 1️⃣ 加载 Dlib 面部检测 & 情绪识别（PyTorch）
# -------------------------------

# 获取 `shape_predictor_68_face_landmarks.dat` 的路径
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"❌ 找不到 {PREDICTOR_PATH}，请确保文件已下载！")

# 加载 Dlib 预训练模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 使用 PyTorch 预训练 ResNet 进行情绪分类
emotion_model = models.resnet18(pretrained=True)  # 改成更轻量的 ResNet18
emotion_model.fc = torch.nn.Linear(512, 7)  # 7 类情绪
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
# 2️⃣ 面部特征提取（OpenCV + Dlib）
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
# 3️⃣ 生成歌词（Ollama / Gemma:2b）
# -------------------------------
def generate_lyrics(facial_features):
    """使用 Ollama `gemma:2b` 生成歌词"""
    prompt = f"A poetic song about a person with {facial_features['face_shape']} face and {facial_features['skin_color']} skin."
    
    # 用 `gemma:2b` 生成歌词
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    
    return response['message']['content']

# -------------------------------
# 4️⃣ 生成旋律（PyTorch 版音乐生成）
# -------------------------------
def generate_melody(emotion):
    """使用 PyTorch 生成旋律"""
    sample_rate = 16000
    melody_length = 4  # 生成 4 秒音频

    # 生成随机音符频率
    freqs = {
        "happy": 440,  # A4
        "sad": 220,  # A3
        "angry": 330,  # E4
        "neutral": 262,  # C4
        "surprise": 523,  # C5
    }
    
    frequency = freqs.get(emotion, 262)
    time = torch.linspace(0, melody_length, steps=melody_length * sample_rate)
    melody_wave = 0.5 * torch.sin(2 * np.pi * frequency * time)

    melody_path = "melody.wav"
    torchaudio.save(melody_path, melody_wave.unsqueeze(0), sample_rate)
    
    return melody_path

# -------------------------------
# 5️⃣ AI 歌曲合成（DiffSinger）
# -------------------------------
def synthesize_song(lyrics, melody_path):
    """调用 DiffSinger 生成歌曲"""
    lyrics_path = "lyrics.txt"
    with open(lyrics_path, "w") as f:
        f.write(lyrics)

    output_wav = "output.wav"
    diff_singer_cmd = f"python /Users/xiangxiaoxin/Documents/GitHub/FaceTune/DiffSinger/inference/ds_acoustic.py --text {lyrics_path} --midi {melody_path} --output {output_wav}"


    if os.system(diff_singer_cmd) != 0:
        raise RuntimeError("❌ DiffSinger 运行失败，请检查是否已正确安装！")
    
    return output_wav

# -------------------------------
# 6️⃣ Gradio 界面
# -------------------------------
def process_image(image):
    """完整的 AI 音乐生成流程"""
    cv2.imwrite("input.jpg", image)
    emotion = detect_emotion("input.jpg")
    features = extract_facial_features("input.jpg")
    lyrics = generate_lyrics(features)
    melody = generate_melody(emotion)
    song = synthesize_song(lyrics, melody)
    
    return lyrics, melody, song

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=["text", "file", "file"],
    title="AI 歌曲生成器",
    description="上传一张照片，AI 将根据你的面部特征生成一首歌曲 🎵"
)

if __name__ == "__main__":
    interface.launch()
