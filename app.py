import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans  # 颜色提取
import ollama  # 歌词生成

# -------------------------------
# 1️⃣ 识别绘画内容 (CLIP + BLIP)
# -------------------------------
model, preprocess, tokenizer = open_clip.create_model_and_transforms("ViT-B/32", pretrained="laion2b_s34b_b79k")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ 初始化 BLIP（用于生成具体的画面描述）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def ensure_pil_image(image):
    """确保 `image` 是 `PIL.Image` 类型，防止 `list` 类型错误"""
    if isinstance(image, dict) and "composite" in image:
        image = Image.fromarray(np.array(image["composite"], dtype=np.uint8))
    elif isinstance(image, list):
        print("📷 image 是 list，转换为 NumPy 数组")
        image = np.array(image, dtype=np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"❌ 错误: image 类型 {type(image)} 不是 PIL.Image")

    return image.convert("RGB")

def extract_visual_features(image):
    """提取画面风格关键词（颜色、线条）"""
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # **颜色风格**
    kmeans = KMeans(n_clusters=3, random_state=0).fit(image_np.reshape(-1, 3))
    colors = kmeans.cluster_centers_.astype(int)
    warm_ratio = sum(1 for c in colors if c[0] > 150 and c[2] < 100) / 3
    dark_ratio = sum(1 for c in colors if sum(c) < 200) / 3
    color_desc = "温暖而充满活力" if warm_ratio > 0.5 else "深沉而神秘" if dark_ratio > 0.5 else "色彩和谐"

    # **线条感觉**
    edges = cv2.Canny(gray, 50, 150)
    line_desc = "线条流畅而自由" if np.count_nonzero(edges) > 10000 else "简洁而富有表现力"

    return f"{color_desc}，{line_desc}"

def describe_image_with_blip(image):
    """使用 BLIP 生成画面描述"""
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption = blip_model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)

def analyze_painting(image):
    """生成画面描述"""

    # ✅ **彻底修复 `image` 类型问题**
    image = ensure_pil_image(image)
    print(f"✅ 转换后 image 类型: {type(image)}")

    # **转换为 Tensor**
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # **使用 BLIP 生成画面描述**
    blip_description = describe_image_with_blip(image)

    # **CLIP 生成情绪关键词**
    descriptions = ["自由而超现实", "梦幻而奇妙", "充满活力", "神秘而深邃", "抽象而富有张力"]
    text_tokens = tokenizer(descriptions).to(device)
    
    with torch.no_grad():
        similarity = (model.encode_image(image_tensor) @ model.encode_text(text_tokens).T).softmax(dim=-1)

    clip_keyword = descriptions[similarity.argmax().item()]
    visual_keywords = extract_visual_features(image)

    return f"{blip_description}，{clip_keyword}，{visual_keywords}"

# -------------------------------
# 2️⃣ 生成歌词 (Ollama / Gemma:2b)
# -------------------------------
def generate_lyrics(painting_description):
    """根据画面描述生成诗意歌词"""
    
    prompt = f"""
    Write a poetic song inspired by this description:
    "{painting_description}"
    
    - Capture the **emotions** of the scene rather than describing it directly.
    - Use **imagery and symbolism** to create a story inspired by the painting.
    - The song should feel like a **mystical journey**, **a lonely adventure**, or **a dreamy reflection**.
    - Avoid generic words like "masterpiece" or "paintbrush". Instead, use metaphors related to art, light, and nature.
    
    Suggested format:
    
    **[Verse 1]**  
    Set the mood with visual imagery and emotional depth.  
    Introduce a **mystical character** (a lost wolf, a wandering artist, a floating soul).  
    
    **[Chorus]**  
    A repeated poetic line that captures the essence of the song.  
    
    **[Verse 2]**  
    Expand on the emotional journey, using **contrast and tension**.  
    
    Examples of poetic styles:  
    - Dreamlike and surreal (e.g., "a golden thread weaves through the sky")  
    - Mysterious and melancholic (e.g., "shadows whisper forgotten names")  
    - Soft and reflective (e.g., "memories drift like paper boats on water")  
    
    **Write in a loose poetic structure, prioritizing storytelling over rhyme.**  
    """

    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    lyrics = response['message']['content']
    
    return format_lyrics(lyrics)

def format_lyrics(lyrics):
    """优化歌词格式，使其更美观"""
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
