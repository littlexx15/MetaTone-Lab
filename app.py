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
    # 去掉或注释掉这行： st.write(f"✅ Using device: {device}")

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
    生成的描述仅打印在终端。
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
    根据图像描述生成诗意歌词，要求内容丰富、节奏流畅，避免重复，并具备清晰的歌曲结构。
    """
    prompt = f"""
    Write a structured poetic song inspired by the following description:
    "{painting_description}"
    
    **Structure:** The song must include [Verse], [Chorus], and optionally [Bridge].  
    **Theme:** Capture deep emotions, vivid imagery, and a dynamic sense of movement.  
    **Variation:** Each section should introduce new elements, avoiding repetitive phrases.  
    **Rhythm & Flow:** Keep lines concise, naturally rhythmic, and easy to sing.  
    **Contrast:** Verses should be introspective and descriptive, while the chorus should be impactful, emotionally intense, and memorable.  
    **Musicality:** Ensure a lyrical structure that fits well with a melody, possibly incorporating rhyme or rhythmic elements.  
    **Emotional Progression:** The song should build up, creating tension and resolution within its narrative.  
    """
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    lyrics = response["message"]["content"]
    return format_lyrics(lyrics)

# -----------------------------------------
# 4️⃣ 核心函数：生成歌曲名称
# -----------------------------------------
def generate_song_title(painting_description):
    """
    根据图像描述生成一个简洁而富有诗意的歌曲名称。
    """
    prompt = f"""
    Based on the following description:
    "{painting_description}"
    
    Provide a concise, creative, and poetic song title. Only output the title.
    """
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    song_title = response["message"]["content"]
    return song_title.strip()

# -----------------------------------------
# 5️⃣ 辅助函数：格式化歌词
# -----------------------------------------
def format_lyrics(lyrics):
    """
    简单的格式化函数，将生成的歌词每行首字母大写，
    并去除多余空行。
    """
    lines = lyrics.split("\n")
    formatted_lines = [line.strip().capitalize() for line in lines if line.strip()]
    return "\n".join(formatted_lines)

# -----------------------------------------
# 6️⃣ Streamlit 界面
# -----------------------------------------
st.title("MetaTone Lab")  # 将标题改为 "MetaTone Lab"
st.write("在画布上自由绘画，点击“生成歌词”后即可获得图像描述（仅打印在终端）、歌曲名称和诗意歌词。")

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
        # 使用 BLIP base 生成图像描述（终端打印）
        painting_description = describe_image_with_blip(image)
        # 基于描述生成歌曲名称
        song_title = generate_song_title(painting_description)
        # 基于描述生成歌词
        lyrics = generate_lyrics(painting_description)

        st.subheader("🎵 歌曲名称")
        st.write(song_title)
        st.subheader("🎶 生成的歌词")
        st.write(lyrics)
    else:
        st.error("请先在画布上绘制内容！")
