import os
import cv2
import dlib
import torch
import numpy as np
import torchaudio
from torchvision import models, transforms
import ollama
import gradio as gr
from TTS.api import TTS

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
# 3️⃣ 生成歌词（Ollama / Gemma:2b）
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
# 4️⃣ 生成旋律（PyTorch 版音乐生成）
# -------------------------------
def generate_melody(emotion):
    """确保音频至少 2 秒"""
    sample_rate = 22050  # 确保采样率够高
    melody_length = 2  # 至少 2 秒
    
    freqs = {
        "happy": 440,
        "sad": 220,
        "angry": 330,
        "neutral": 262,
        "surprise": 523,
    }
    
    frequency = freqs.get(emotion, 262)
    time = torch.linspace(0, melody_length, steps=int(melody_length * sample_rate))  # 修正 time 计算
    melody_wave = 0.5 * torch.sin(2 * np.pi * frequency * time)

    melody_path = "melody.wav"
    torchaudio.save(melody_path, melody_wave.unsqueeze(0), sample_rate)
    
    return melody_path


# -------------------------------
# 5️⃣ 使用 FastPitch 进行歌曲合成
# -------------------------------
def synthesize_song(lyrics, melody_path):
    """使用 FastPitch 进行语音合成"""
    
    tts = TTS("tts_models/en/ljspeech/fast_pitch")  # ✅ 改用 FastPitch，速度更快
    output_wav = "output.wav"
    
    # 生成语音并加快语速，防止声音拉长
    tts.tts_to_file(text=lyrics, file_path=output_wav, speed=1.1, max_decoder_steps=500)

    return output_wav



# -------------------------------
# 6️⃣ Gradio 界面（在线播放）
# -------------------------------
def process_image(image):
    """完整的 AI 音乐生成流程"""
    cv2.imwrite("input.jpg", image)
      
    # 检测情绪
    emotion = detect_emotion("input.jpg")
    print(f"🧐 识别的情绪：{emotion}")  # ✅ 打印情绪识别结果

    # 提取面部特征
    features = extract_facial_features("input.jpg")
    
    # 生成歌词（结合面部特征 & 情绪）
    lyrics = generate_lyrics(features, emotion)
    
    # 生成旋律（基于情绪）
    melody = generate_melody(emotion)
    
    # 合成歌曲
    song = synthesize_song(lyrics, melody)
    
    return lyrics, melody, song


interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        "text",  # 歌词文本
        gr.Audio(type="filepath", format="wav"),  # 🎵 旋律（在线播放）
        gr.Audio(type="filepath", format="wav")   # 🎤 生成的歌曲（在线播放）
    ],
    title="AI 歌曲生成器",
    description="上传一张照片，AI 将根据你的面部特征生成一首歌曲 🎵"
)

if __name__ == "__main__":
    print("🚀 Python 运行成功！")
    interface.launch()
