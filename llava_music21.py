import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import random
import music21
import tempfile

from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

# -------------------------------
# 0️⃣ 页面布局与全局样式
# -------------------------------
st.set_page_config(
    page_title="MetaTone Lab",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;
        margin: auto;
    }
    h1 {
        text-align: center;
        font-size: 36px !important;
        margin-bottom: 0.2em;
    }
    .subheader-text {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 0.6em;
        margin-top: 0.2em;
    }
    .song-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 0.5em;
        margin-bottom: 0.5em;
    }
    .lyrics-container {
        height: 500px;
        overflow-y: auto;
        padding-right: 1em;
        margin-top: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .lyrics-container p {
        line-height: 1.6;
        margin-bottom: 0.8em;
        margin-left: 0.5em;
        margin-right: 0.5em;
    }
    .stButton {
        margin-top: 1em;
        margin-bottom: 1em;
    }
    div[data-baseweb="slider"] {
        width: 500px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>MetaTone Lab</h1>", unsafe_allow_html=True)

# -------------------------------
# 1️⃣ 生成歌词
# -------------------------------
def generate_lyrics_with_ollama(image: Image.Image) -> str:
    """
    将绘制的图像保存为临时文件，然后调用 llava:7b 模型生成结构化歌词。
    """
    temp_path = create_temp_file(image)
    prompt = """
You are a creative songwriting assistant.
Please look at the image I provide and write a structured poetic song inspired by the visual content.

**Requirements**:
1. The song must include [Verse], [Chorus], and optionally [Bridge].
2. Capture deep emotions, vivid imagery, and a dynamic sense of movement.
3. Each section should introduce new elements, avoiding repetitive phrases.
4. Keep lines concise, naturally rhythmic, and easy to sing.
5. Verses should be introspective and descriptive, while the chorus should be impactful, emotionally intense, and memorable.
6. Build emotional tension and resolution within the narrative.

Now here is the image:
    """
    stream = analyze_image_file(
        image_file=temp_path,
        model="llava:7b",
        user_prompt=prompt
    )
    parsed = stream_parser(stream)
    lyrics = "".join(parsed).strip()
    # 如果模型返回的字符串首尾有双引号，则去掉
    lyrics = lyrics.strip('"')
    return lyrics

# -------------------------------
# 2️⃣ 生成歌曲标题
# -------------------------------
def generate_song_title(image: Image.Image) -> str:
    """
    将绘制的图像保存为临时文件，然后调用 llava:7b 模型生成歌曲标题。
    """
    temp_path = create_temp_file(image)
    prompt = """
Provide a concise, creative, and poetic song title. Only output the title, with no extra words or disclaimers.
    """
    stream = analyze_image_file(
        image_file=temp_path,
        model="llava:7b",
        user_prompt=prompt
    )
    parsed = stream_parser(stream)
    title = "".join(parsed).strip()
    title = title.strip('"')
    return title

# -------------------------------
# 3️⃣ 格式化歌词
# -------------------------------
def format_text(text: str) -> str:
    """
    去除多余空行，并保证每行首字母大写。
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [l[0].upper() + l[1:] if l else "" for l in lines]
    return "\n\n".join(lines)

# -------------------------------
# 4️⃣ 为歌词生成随机旋律的 MIDI
# -------------------------------
def generate_random_melody(lyrics: str) -> bytes:
    """
    将多行歌词拆分，每行分配一个随机音高，生成 MIDI 文件并返回其二进制内容。
    """
    # 1. 拆分歌词成行
    lines = [l.strip() for l in lyrics.split("\n") if l.strip()]

    # 2. 创建 music21 的 Stream
    s = music21.stream.Stream()

    # 3. 定义一个音阶，用来随机挑选音高
    scale_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]

    for line in lines:
        # 为这一行歌词分配一个随机音高
        pitch = random.choice(scale_notes)
        # 创建一个 note，时长先固定为 1.0 四分音符
        n = music21.note.Note(pitch, quarterLength=1.0)
        # 将这一行歌词放到 note 的 lyric 字段
        n.lyric = line
        s.append(n)

    # 4. 写到临时文件中
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi_path = tmp.name
    s.write("midi", fp=midi_path)

    # 5. 读出二进制内容并返回
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    return midi_bytes

# -------------------------------
# 5️⃣ 主布局：左侧绘画，右侧生成结果
# -------------------------------
col_left, col_right = st.columns([1.4, 1.6], gap="medium")

# 左侧：绘画区域
with col_left:
    st.markdown("<div class='subheader-text'>在这里画画</div>", unsafe_allow_html=True)
    st.write("选择画笔颜色和笔刷大小，自由绘制创意画面。")
    brush_color = st.color_picker("画笔颜色", value="#000000")
    brush_size = st.slider("画笔大小", 1, 50, value=5)
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

# 右侧：生成结果区域
with col_right:
    st.markdown("<div class='subheader-text'>生成结果</div>", unsafe_allow_html=True)
    st.write("完成绘画后，点击下方按钮生成歌曲标题与歌词。")

    if st.button("🎶 生成歌曲"):
        if canvas_result.image_data is not None:
            # 将绘制结果转换为 PIL Image 对象
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
            
            # 调用 llava:7b 生成歌曲标题和歌词
            song_title = generate_song_title(image)
            raw_lyrics = generate_lyrics_with_ollama(image)
            
            # 对歌词进行格式化
            lyrics = format_text(raw_lyrics)
            
            # 展示结果
            st.markdown("**歌曲标题：**", unsafe_allow_html=True)
            st.markdown(f"<div class='song-title'>{song_title}</div>", unsafe_allow_html=True)

            st.markdown("**生成的歌词：**", unsafe_allow_html=True)
            lyrics_html = lyrics.replace("\n", "<br>")
            st.markdown(
                f"<div class='lyrics-text lyrics-container'><p>{lyrics_html}</p></div>",
                unsafe_allow_html=True
            )

            # 额外：为歌词生成一个简单随机旋律的 MIDI
            midi_bytes = generate_random_melody(lyrics)
            st.markdown("#### 随机旋律 MIDI 文件")
            st.download_button(
                label="下载随机旋律 MIDI",
                data=midi_bytes,
                file_name="random_melody.mid",
                mime="audio/midi"
            )

        else:
            st.error("请先在左侧画布上绘制内容！")
