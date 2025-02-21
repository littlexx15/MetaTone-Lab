import torch
import numpy as np
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama  # 用于歌词生成
from streamlit_drawable_canvas import st_canvas

# -----------------------------------------
# 1️⃣ 缓存或初始化模型
# -----------------------------------------
@st.cache_resource  # 使用 Streamlit 缓存，避免每次重跑都加载模型
def load_blip_large_model():
    """
    加载 blip-image-captioning-large 模型和处理器。
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    return device, processor, blip_model

device, processor, blip_model = load_blip_large_model()

# -----------------------------------------
# 2️⃣ 核心函数：描述图像
# -----------------------------------------
def describe_image_with_blip(image):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(
            **inputs,
            max_length=120,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=1.0,
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    caption_str = processor.decode(caption_ids[0], skip_special_tokens=True)
    print(f"[BLIP Large 描述] {caption_str}")
    return caption_str

def generate_lyrics(painting_description):
    """
    根据画面描述生成诗意歌词。
    """
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
    
    **Write in a loose poetic structure, prioritizing storytelling over rhyme.**
    """

    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    lyrics = response["message"]["content"]
    return format_lyrics(lyrics)

def format_lyrics(lyrics):
    """简单的格式化，将每行首字母大写"""
    lines = lyrics.split("\n")
    formatted_lines = [line.strip().capitalize() for line in lines if line.strip()]
    return "\n".join(formatted_lines)

# -----------------------------------------
# 3️⃣ Streamlit 界面
# -----------------------------------------
st.title("🎨 AI 绘画歌词生成器")
st.write("在画布上自由绘画，点击“生成歌词”后即可获得对画面的诗意描述与歌词 🎵")

# 颜色选择器和笔刷大小
brush_color = st.color_picker("选择画笔颜色", value="#000000")
brush_size = st.slider("画笔大小", 1, 50, value=5)

# 画布
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=brush_size,
    stroke_color=brush_color,
    background_color="white",
    update_streamlit=True,
    width=512,
    height=512,
    drawing_mode="freedraw",
    key="canvas",
)

# 生成歌词
if st.button("🎶 生成歌词"):
    if canvas_result.image_data is not None:
        # 将画布数据转为 PIL Image
        image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")

        # 使用 BLIP large 生成描述
        painting_description = describe_image_with_blip(image)

        # 基于描述生成歌词
        lyrics = generate_lyrics(painting_description)

        st.subheader("🖼 识别的绘画内容")
        st.write(painting_description)

        st.subheader("🎶 生成的歌词")
        st.write(lyrics)
    else:
        st.error("请先在画布上绘制内容！")
