import os
import cv2
import torch
import numpy as np
import gradio as gr
from torchvision import transforms
from PIL import Image
import ollama
import open_clip

# -------------------------------
# 1️⃣ 识别绘画内容 (CLIP)
# -------------------------------
model, preprocess, tokenizer = open_clip.create_model_and_transforms("ViT-B/32", pretrained="laion2b_s34b_b79k")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def analyze_painting(image):
    """使用 CLIP 识别绘画内容，生成描述"""

    print(f"📷 image 类型: {type(image)}")  # 打印 image 的类型
    if isinstance(image, dict):
        print(f"📷 image.keys(): {image.keys()}")  # 查看字典键

    # ✅ 处理 Gradio Sketchpad 传入的 dict 数据
    if isinstance(image, dict):  
        if "composite" in image:  # Sketchpad 返回的数据结构包含 'composite'
            image = image["composite"]  # 提取 composite 数据
            print(f"📷 提取 composite 后 image 类型: {type(image)}")  
        else:
            raise ValueError(f"image 字典中没有 'composite' 键，实际内容: {image.keys()}")

    # ✅ 处理 list 类型，转换为 NumPy 数组
    if isinstance(image, list):
        print("📷 image 是 list，尝试转换为 NumPy 数组")
        image = np.array(image, dtype=np.uint8)

    # ✅ 确保 image 是 NumPy 数组，避免 torchvision 报错
    if not isinstance(image, np.ndarray):
        raise TypeError(f"转换失败，image 仍然是 {type(image)}，应为 NumPy 数组")

    print(f"📷 确保 image 现在是 NumPy 数组: {type(image)}")
    
    image = Image.fromarray(image)  # 转换成 PIL.Image
    print(f"📷 转换为 PIL.Image 后 image 类型: {type(image)}")  
    image = image.convert("RGB")  # 转换为 RGB 格式
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    return "测试通过"

    
    # CLIP 预定义的文本描述类别
    descriptions = [
        "a surreal painting",
        "an abstract artwork",
        "a fantasy scene",
        "a futuristic cityscape",
        "a dreamy landscape",
        "a dark, melancholic scene",
        "a bright and colorful painting",
        "a mysterious, eerie painting"
    ]
    text_tokens = tokenizer(descriptions).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_match = descriptions[similarity.argmax().item()]
    
    return best_match

# -------------------------------
# 2️⃣ 生成歌词 (Ollama / Gemma:2b)
# -------------------------------
def generate_lyrics(painting_description):
    """根据绘画描述生成诗意歌词"""
    prompt = f"""
    Write a poetic song inspired by {painting_description}.
    The song should evoke emotions and create vivid imagery.
    Use metaphor, symbolism, and a storytelling format:
    - [Verse 1] Introduce the scene inspired by the painting.
    - [Chorus] A memorable refrain that captures the song's essence.
    - [Verse 2] Expand on the narrative, adding depth and contrast.

    Example of the desired style:
    - Like Bob Dylan or Leonard Cohen, poetic and evocative lyrics.
    - Follow a loose rhyme scheme (AABB or ABAB) but prioritize storytelling.
    """
    
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    lyrics = response['message']['content']
    return format_lyrics(lyrics)

def format_lyrics(lyrics):
    """优化歌词格式"""
    lines = lyrics.split("\n")
    formatted_lines = [line.strip().capitalize() for line in lines if line.strip()]
    return "\n".join(formatted_lines)

# -------------------------------
# 3️⃣ Gradio 界面 (绘画输入)
# -------------------------------
def process_painting(image):
    """完整的 AI 歌词生成流程"""
    painting_description = analyze_painting(image)
    print(f"🖼 识别的绘画风格：{painting_description}")
    
    # 生成歌词
    lyrics = generate_lyrics(painting_description)
    
    return f"🎨 识别的绘画风格：{painting_description}\n🎶 生成的歌词：\n{lyrics}"

interface = gr.Interface(
    fn=process_painting,
    inputs=gr.Sketchpad(),  # ✅ 直接去掉 output_mode
    outputs="text",
    title="AI 绘画歌词生成器",
    description="在画布上绘制一幅画，AI 将根据内容生成一首歌词 🎵",
)

if __name__ == "__main__":
    print("🚀 Python 运行成功！")
    interface.launch()  # ✅ 正确写法
