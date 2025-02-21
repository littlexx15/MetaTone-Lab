import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans  # 颜色提取
import ollama  # 歌词生成
from streamlit_drawable_canvas import st_canvas  # 替换 Gradio 画布

# -------------------------------
# 1️⃣ 缓存加载模型，避免重复加载耗时
# -------------------------------
@st.cache_resource
def load_models_and_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ 使用设备: {device}")

    # 加载 OpenCLIP 模型
    model, preprocess, tokenizer = open_clip.create_model_and_transforms(
        "ViT-L/14", pretrained="openai"
    )
    model.to(device)

    # 加载 BLIP 模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    return device, model, preprocess, tokenizer, processor, blip_model

device, model, preprocess, tokenizer, processor, blip_model = load_models_and_device()

# -------------------------------
# 2️⃣ 功能函数
# -------------------------------
def ensure_pil_image(image):
    """确保 image 是 PIL.Image 类型"""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")

def extract_visual_features(image):
    """提取画面风格关键词（颜色、线条）"""
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 颜色风格
    kmeans = KMeans(n_clusters=3, random_state=0).fit(image_np.reshape(-1, 3))
    colors = kmeans.cluster_centers_.astype(int)
    warm_ratio = sum(1 for c in colors if c[0] > 150 and c[2] < 100) / 3
    dark_ratio = sum(1 for c in colors if sum(c) < 200) / 3
    color_desc = "温暖而充满活力" if warm_ratio > 0.5 else "深沉而神秘" if dark_ratio > 0.5 else "色彩和谐"

    # 线条感觉
    edges = cv2.Canny(gray, 50, 150)
    line_desc = "线条流畅而自由" if np.count_nonzero(edges) > 10000 else "简洁而富有表现力"

    return f"{color_desc}，{line_desc}"

def describe_image_with_blip(image):
    """使用 BLIP 生成更丰富的画面描述"""
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption = blip_model.generate(**inputs, max_length=50, do_sample=True, temperature=0.9)
    return processor.decode(caption[0], skip_special_tokens=True)

def analyze_painting(image):
    """生成画面描述"""
    image = ensure_pil_image(image)
    print(f"✅ 转换后 image 类型: {type(image)}")

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    blip_description = describe_image_with_blip(image)

    descriptions = ["自由而超现实", "梦幻而奇妙", "充满活力", "神秘而深邃", "抽象而富有张力"]
    text_tokens = open_clip.tokenize(descriptions).to(device)
    
    with torch.no_grad():
        similarity = (model.encode_image(image_tensor) @ model.encode_text(text_tokens).T).softmax(dim=-1)

    clip_keyword = descriptions[similarity.argmax().item()]
    visual_keywords = extract_visual_features(image)

    return f"{blip_description}，{clip_keyword}，{visual_keywords}"

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
    """优化歌词格式"""
    lines = lyrics.split("\n")
    formatted_lines = [line.strip().capitalize() for line in lines if line.strip()]
    return "\n".join(formatted_lines)

# -------------------------------
# 3️⃣ Streamlit 界面（集成控件和画布）
# -------------------------------
st.title("🎨 AI 绘画歌词生成器")
st.write("直接在下方调整颜色和笔刷大小，然后开始绘画，AI 将生成歌词 🎵")

# 将颜色选择和笔刷大小控件放在画布上方
brush_color = st.color_picker("选择画笔颜色", value="#000000")
brush_size = st.slider("画笔大小", 1, 50, value=5)

# 提示：每次调整颜色或笔刷大小会重载画布，可能导致当前绘图内容被清空！
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=brush_size,
    stroke_color=brush_color,
    background_color="white",
    update_streamlit=True,  # 开启实时更新，实时传输每一笔
    width=512,
    height=512,
    drawing_mode="freedraw",
    key="canvas"  # 固定 key 尽量保留状态
)

if st.button("🎶 生成歌词"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
        painting_description = analyze_painting(image)
        lyrics = generate_lyrics(painting_description)
        st.subheader("🎨 识别的绘画风格")
        st.write(painting_description)
        st.subheader("🎶 生成的歌词")
        st.write(lyrics)
    else:
        st.error("请先在画布上绘制内容！")
