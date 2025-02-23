import torch
import numpy as np
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama  # 用于歌词生成
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# 0️⃣ 页面布局与全局样式
# -------------------------------
st.set_page_config(
    page_title="MetaTone Lab",
    layout="wide",  # 宽屏布局
)

# 全局CSS，控制字体大小、标题居中、歌词滚动等
st.markdown(
    """
    <style>
    /* 调整整页左右边距，减少空白，改为1200px以适应普通屏幕 */
    .main .block-container {
        max-width: 1200px;
        margin: auto;
    }
    /* 顶部主标题：居中 + 大字号 */
    h1 {
        text-align: center;
        font-size: 36px !important;
        margin-bottom: 0.2em;
    }
    /* 二级标题、说明等：稍微大些 */
    .subheader-text {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 0.6em;
        margin-top: 0.2em;
    }
    /* 歌曲名称：加大一点字号 */
    .song-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 0.5em;
        margin-bottom: 0.5em;
    }
    /* 歌词容器：固定高度 500px + 滚动条，和画布保持一致 */
    .lyrics-container {
        height: 500px;
        overflow-y: auto;
        padding-right: 1em;
        margin-top: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    /* 歌词文本的行距、段落间距 */
    .lyrics-container p {
        line-height: 1.6;
        margin-bottom: 0.8em;
        margin-left: 0.5em;
        margin-right: 0.5em;
    }
    /* 调整按钮的间距 */
    .stButton {
        margin-top: 1em;
        margin-bottom: 1em;
    }
    /* 强制使slider的宽度=500px（与画布宽度相同） */
    div[data-baseweb="slider"] {
        width: 500px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 顶部主标题
st.markdown("<h1>MetaTone Lab</h1>", unsafe_allow_html=True)

# -------------------------------
# 1️⃣ 缓存或初始化模型：BLIP base
# -------------------------------
@st.cache_resource
def load_blip_base_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return device, processor, blip_model

device, processor, blip_model = load_blip_base_model()

# -------------------------------
# 2️⃣ 描述图像
# -------------------------------
def describe_image_with_blip(image):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(
            **inputs,
            max_length=80,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    caption_str = processor.decode(caption_ids[0], skip_special_tokens=True)
    print(f"[BLIP Base 描述] {caption_str}")
    return caption_str

# -------------------------------
# 3️⃣ 生成歌词
# -------------------------------
def generate_lyrics(painting_description):
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

# -------------------------------
# 4️⃣ 生成歌曲名称
# -------------------------------
def generate_song_title(painting_description):
    prompt = f"""
    Based on the following description:
    "{painting_description}"
    
    Provide a concise, creative, and poetic song title. Just output the title, with no extra words or disclaimers.
    """
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    song_title = response["message"]["content"]
    return song_title.strip()

# -------------------------------
# 5️⃣ 格式化歌词
# -------------------------------
def format_lyrics(lyrics):
    lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    lines = [l[0].upper() + l[1:] if l else "" for l in lines]
    joined = "\n\n".join(lines)
    return joined

# -------------------------------
# 6️⃣ 主布局
# -------------------------------
col_left, col_right = st.columns([1.4, 1.6], gap="medium")

# 左侧：绘画区域
with col_left:
    st.markdown("<div class='subheader-text'>在这里画画</div>", unsafe_allow_html=True)
    st.write("选择画笔颜色和笔刷大小，自由绘制创意画面。")

    brush_color = st.color_picker("画笔颜色", value="#000000")
    brush_size = st.slider("画笔大小", 1, 50, value=5)

    # 画布 500×500
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=brush_size,
        stroke_color=brush_color,
        background_color="white",
        update_streamlit=True,
        width=550,
        height=550,
        drawing_mode="freedraw",
        key="canvas",
    )

# 右侧：生成结果
with col_right:
    st.markdown("<div class='subheader-text'>生成结果</div>", unsafe_allow_html=True)
    st.write("完成绘画后，点击下方按钮生成歌曲名称与歌词。")

    if st.button("🎶 生成歌词"):
        if canvas_result.image_data is not None:
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
            # 生成图像描述（仅在终端打印）
            painting_description = describe_image_with_blip(image)
            # 生成歌曲名称
            song_title = generate_song_title(painting_description)
            # 生成歌词
            lyrics = generate_lyrics(painting_description)

            st.markdown("**歌曲名称：**", unsafe_allow_html=True)
            st.markdown(f"<div class='song-title'>{song_title}</div>", unsafe_allow_html=True)

            st.markdown("**生成的歌词：**", unsafe_allow_html=True)
            # 将换行符替换成 <br>
            lyrics_html = lyrics.replace("\n", "<br>")
            # 高度固定 500px，和画布相同
            st.markdown(
                f"<div class='lyrics-text lyrics-container'><p>{lyrics_html}</p></div>",
                unsafe_allow_html=True
            )
        else:
            st.error("请先在左侧画布上绘制内容！")
