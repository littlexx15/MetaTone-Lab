import os
import cv2
import dlib
import torch
import numpy as np
import gradio as gr
from torchvision import models, transforms
import ollama
from deepface import DeepFace

# -------------------------------
# 1️⃣ 面部检测 & 情绪识别
# -------------------------------
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"❌ 找不到 {PREDICTOR_PATH}，请确保文件已下载！")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def detect_emotion(image_path):
    """使用 DeepFace 进行情绪识别"""
    analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
    return analysis[0]['dominant_emotion']

# -------------------------------
# 2️⃣ 面部特征提取
# -------------------------------
def extract_facial_features(image_path):
    """使用 OpenCV 和 dlib 提取面部特征"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    features = {
        "face_shape": "unknown",
        "skin_color": "unknown",
        "hair_color": "unknown",
        "facial_hair": "none",
        "nose_shape": "unknown",
        "glasses": "none",
        "symmetry": "unknown",
        "hat": "none"
    }

    for face in faces:
        landmarks = predictor(gray, face)
        features["face_shape"] = "oval" if landmarks.part(0).x < landmarks.part(16).x else "round"
        avg_color = np.mean(img, axis=(0, 1))
        features["skin_color"] = "light" if avg_color[2] > 160 else "dark"
        features["hair_color"] = "brown" if avg_color[0] > 80 else "black"
        features["facial_hair"] = "beard" if np.mean(gray[landmarks.part(8).y:landmarks.part(30).y, :]) < 90 else "none"
        features["glasses"] = "yes" if np.mean(gray[landmarks.part(36).y:landmarks.part(45).y, :]) < 50 else "none"
        features["hat"] = "yes" if np.mean(gray[:landmarks.part(19).y, :]) < 60 else "none"
    
    return features

# -------------------------------
# 3️⃣ 生成歌词 (Ollama / Gemma:2b)
# -------------------------------
def generate_lyrics(facial_features, emotion):
    """结合面部特征和情绪生成优化后的歌词"""
    
    prompt = f"""
    Write a poetic song inspired by folk storytelling, rich in imagery and emotion.
    The song is about a person with {facial_features['face_shape']} face, {facial_features['skin_color']} skin, {facial_features['hair_color']} hair, and wearing {facial_features['glasses']}.
    They are feeling {emotion}. 
    Use metaphor, symbolism, and vivid descriptions to enhance the lyrics.

    Structure the lyrics in a storytelling format:
    - [Verse 1] Introduce the scene and the main character's emotions.
    - [Chorus] A memorable, poetic refrain that captures the song's essence.
    - [Verse 2] Develop the narrative, adding depth and contrast.

    Example of the desired style:
    - Like Bob Dylan or Leonard Cohen, the lyrics should feel poetic, thoughtful, and evocative.
    - Ensure the lyrics follow a loose rhyme scheme (AABB or ABAB) but prioritize storytelling over strict rhyming.
    """
    
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    lyrics = response['message']['content']

    # 确保歌词格式良好
    lyrics = format_lyrics(lyrics)

    # 避免歌词过短，增加一些诗意的结尾
    if len(lyrics.split()) < 15:
        lyrics += "\nAnd so the night fades into longing, as the echoes of love remain."

    return lyrics

def format_lyrics(lyrics):
    """优化歌词格式，使其更整齐、更有诗意"""
    lines = lyrics.split("\n")
    formatted_lines = [line.strip().capitalize() for line in lines if line.strip()]
    return "\n".join(formatted_lines)


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

