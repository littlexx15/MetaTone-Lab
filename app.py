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

# ✅ 初始化 BLIP（用于生成具体的画面描述）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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
    """生成画面描述，不只是分类"""
    if isinstance(image, dict) and "composite" in image:
        image = Image.fromarray(np.array(image["composite"], dtype=np.uint8))

    image = image.convert("RGB")  
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # **使用 BLIP 生成画面描述**
    blip_description = describe_image_with_blip(image)

    # **CLIP 生成情绪词**
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
