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
            max_length=80,
            do_sample=True,
            top_p=0.8,
            top_k=40,
            temperature=0.7,
            num_return_sequences=1,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2
        )# 降低top_p和temperature减少幻想
    caption_str = processor.decode(caption_ids[0], skip_special_tokens=True)
    print(f"[BLIP Large 描述] {caption_str}")
    return caption_str

def generate_lyrics(painting_description):
    """
    根据画面描述生成更具故事性、情感深度和叙事结构的歌词。
    参考了写歌词的进阶技巧：故事讲述、情感共鸣、隐喻明喻、叙事结构、押韵等。
    """
    prompt = f"""
    You are a skilled lyricist and storyteller. Based on the following description, please write a poetic song:
    "{painting_description}"
    
    In this song, apply the following advanced songwriting guidelines:
    1. **Storytelling and Emotional Resonance**: Craft a clear narrative arc that can emotionally engage listeners. 
       - Let the story unfold across verses, building tension or insight before the chorus. 
       - Make sure the emotions are authentic, drawing on personal or universal truths.
    2. **Imagery and Symbolism**: Use vivid imagery, metaphors, and similes to create a mental picture. 
       - Let the visuals from the painting inform symbolic elements or hidden meanings in your lyrics.
    3. **Song Structure**: Organize the song with verses, chorus, and optionally a bridge or pre-chorus.
       - Verses: reveal details of the story or the emotional journey.
       - Chorus: capture the essence or main theme, repeated as a memorable hook.
       - Bridge: provide a moment of reflection, contrast, or a turning point in the narrative.
    4. **Rhyme and Musicality**: Aim for a sense of rhythm and flow. 
       - You can use simple or slant rhymes, but keep them subtle. 
       - Make the words feel naturally musical, even when read aloud.
    5. **Balance with Melody**: Though we don't have an actual melody here, write lyrics that could be easily set to music.
       - Keep lines relatively concise. 
       - Avoid overly dense text that might be hard to sing.
    6. **Focus on the Lyricist's Role**: Remember the importance of the lyricist in shaping the emotional core of a song. 
       - Let the words complement an imaginary melody without overshadowing it.
    7. **Avoid Overused Words**: 
       - Steer clear of generic words like "masterpiece," "paintbrush," or clichés that might cheapen the emotional impact.
    8. **Reference Emotional Context**: If the painting has a certain mood or color palette, let that influence the tone of the song.

    **Additional tips**:
    - Draw on the synergy between composer and lyricist (like Carole King and Gerry Goffin), where the lyrics fit seamlessly with the imagined music.
    - Keep it straightforward yet emotionally impactful, similar to how Goffin’s lyrics were direct but deeply resonant.
    - Focus on capturing the painting’s emotional essence rather than describing it literally.

    **Suggested format**:
    - [Verse 1]: Introduce the setting or main character, setting the tone for the story.
    - [Chorus]: A repeated poetic line or theme that captures the essence of the song.
    - [Verse 2]: Expand on the narrative, show progression or conflict.
    - [Bridge or Pre-Chorus (optional)]: A twist, reflection, or emotional pivot.
    - [Chorus - repeated]

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
