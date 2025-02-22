import torch
import numpy as np
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama  # 用于歌词生成
from streamlit_drawable_canvas import st_canvas

# -----------------------------------------
# 1️⃣ 缓存或初始化模型：BLIP base
# -----------------------------------------
@st.cache_resource
def load_blip_base_model():
    """
    加载 Salesforce/blip-image-captioning-base 模型和处理器，
    用于生成图像描述（基础版本）。
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st.write(f"✅ Using device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    return device, processor, blip_model

device, processor, blip_model = load_blip_base_model()

# -----------------------------------------
# 2️⃣ 核心函数：描述图像（使用 BLIP base）
# -----------------------------------------
def describe_image_with_blip(image):
    """
    使用 Salesforce/blip-image-captioning-base 模型生成图像描述，
    直接输出模型的结果（不做颜色纠正），
    以观察基础模型对颜色和内容的识别效果。
    """
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(
            **inputs,
            max_length=80,       # 允许足够细节描述
            do_sample=False,     # 关闭随机采样，确保确定性输出
            num_beams=5,         # 使用 Beam Search 提高输出质量
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    caption_str = processor.decode(caption_ids[0], skip_special_tokens=True)
    print(f"[BLIP Base 描述] {caption_str}")
    return caption_str

# -----------------------------------------
# 3️⃣ 核心函数：生成歌词
# -----------------------------------------
def generate_lyrics(painting_description):
    """
    根据图像描述生成更具故事性、情感深度和叙事结构的歌词。
    """
    prompt = f"""
    You are a skilled lyricist and storyteller. Based on the following description, please write a poetic song:
    "{painting_description}"
    
    In this song, apply the following advanced songwriting guidelines:
    1. **Storytelling and Emotional Resonance**: Craft a clear narrative arc that can emotionally engage listeners. 
       - Let the story unfold across verses, building tension or insight before the chorus. 
       - Ensure the emotions are authentic, drawing on personal or universal truths.
    2. **Imagery and Symbolism**: Use vivid imagery, metaphors, and similes to create a mental picture. 
       - Let the visuals from the painting inform symbolic elements or hidden meanings in your lyrics.
    3. **Song Structure**: Organize the song with verses, chorus, and optionally a bridge or pre-chorus.
       - Verses: reveal details of the story or the emotional journey.
       - Chorus: capture the main theme, repeated as a memorable hook.
       - Bridge: provide a twist or moment of reflection.
    4. **Rhyme and Musicality**: Aim for a rhythmic flow with subtle or slant rhymes.
    5. **Balance with Melody**: Write lyrics that could be easily set to music, keeping lines concise.
    6. **Focus on Emotional Essence**: Capture the painting's emotional core rather than describing it literally.
    7. **Avoid Clichés**: Steer clear of overused phrases and generic words.
    8. **Reference Color and Mood**: Let the painting’s color palette and mood influence the tone of the song.
    
    **Write in a loose poetic structure, prioritizing storytelling over rigid rhyme.**
    **Ensure the final piece feels cohesive, imaginative, and emotionally resonant.**
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
# 4️⃣ Streamlit 界面
# -----------------------------------------
st.title("🎨 AI 绘画歌词生成器 (BLIP Base)")
st.write("在画布上自由绘画，点击“生成歌词”后即可获得图像描述和诗意歌词。")

brush_color = st.color_picker("选择画笔颜色", value="#000000")
brush_size = st.slider("画笔大小", 1, 50, value=5)

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

if st.button("🎶 生成歌词"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
        # 使用 BLIP base 生成图像描述（无颜色纠正）
        painting_description = describe_image_with_blip(image)
        # 基于描述生成歌词
        lyrics = generate_lyrics(painting_description)

        st.subheader("🖼 识别的绘画内容")
        st.write(painting_description)
        st.subheader("🎶 生成的歌词")
        st.write(lyrics)
    else:
        st.error("请先在画布上绘制内容！")
