import streamlit as st
st.set_page_config(page_title="MetaTone Lab", layout="wide")

import sys
import os
import subprocess
import tempfile
import glob
import shutil
import json
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random

print("Python executable:", sys.executable)

# 在这里用环境变量来取得用户的 Python 和 infer.py 路径
#   - YUE_PYTHON   -> 指向“专门给 yueForWindows 准备的 Python”（可能是某个 conda env 或 venv）
#   - YUE_INFER_PY -> 指向用户下载的 yueForWindows 目录下的 "inference/infer.py"
#   - YUE_CWD      -> 如果一定需要在 yueForWindows 目录下执行，指定它的工作目录
YUE_PYTHON   = os.environ.get("YUE_PYTHON",   "python")  # 如果用户没设置，默认用 "python"
YUE_INFER_PY = os.environ.get("YUE_INFER_PY", "yueForWindows/inference/infer.py") 
YUE_CWD      = os.environ.get("YUE_CWD",      "yueForWindows") 

# =============== 辅助函数 ===============
from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser


# =============== session_state 存储歌词和标题 ===============
if "lyrics" not in st.session_state:
    st.session_state["lyrics"] = None
if "song_title" not in st.session_state:
    st.session_state["song_title"] = None

# =============== 页面样式（仅调用一次）===============
st.markdown(
    """
    <style>
    .main .block-container { max-width: 1200px; margin: auto; }
    h1 { text-align: center; font-size: 36px !important; margin-bottom: 0.2em; }
    .subheader-text { font-size: 20px; font-weight: bold; margin-bottom: 0.6em; margin-top: 0.2em; }
    .song-title { font-size: 24px; font-weight: bold; margin-top: 0.5em; margin-bottom: 0.5em; }
    .lyrics-container { height: 500px; overflow-y: auto; padding-right: 1em; margin-top: 10px; border: 1px solid #ccc; border-radius: 5px; }
    .lyrics-container p { line-height: 1.6; margin-bottom: 0.8em; margin-left: 0.5em; margin-right: 0.5em; }
    .stButton { margin-top: 1em; margin-bottom: 1em; }
    div[data-baseweb="slider"] { width: 500px !important; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1>MetaTone Lab</h1>", unsafe_allow_html=True)

# =============== 1) 生成歌词 (调用 llava:7b) ===============
def generate_lyrics_with_ollama(image: Image.Image) -> str:
    """调用 llava:7b 模型，根据图像生成英文歌词。"""
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
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    lyrics = "".join(parsed).strip()
    return lyrics.strip('"')

# =============== 2) 生成歌曲标题 (调用 llava:7b) ===============
def generate_song_title(image: Image.Image) -> str:
    """调用 llava:7b 模型，为图像生成歌曲标题。"""
    temp_path = create_temp_file(image)
    prompt = """
Provide a concise, creative, and poetic song title. Only output the title, with no extra words or disclaimers.
    """
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    title = "".join(parsed).strip()
    return title.strip('"')

# =============== 3) 格式化歌词 ===============
def format_text(text: str) -> str:
    """去除多余空行，并保证每行首字母大写。"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [l[0].upper() + l[1:] if l else "" for l in lines]
    return "\n\n".join(lines)

# =============== YuE 推理函数 ===============
def yue_infer(lyrics: str) -> bytes:
    """
    使用 YuE（或 YuEGP）的推理脚本从文本生成完整歌曲。
    """
    
    # 1) 创建临时目录
    temp_out_dir = tempfile.mkdtemp()
    
    # 2) 将生成的歌词写入一个临时文件
    lyrics_file = os.path.join(temp_out_dir, "lyrics.txt")
    with open(lyrics_file, "w") as f:
        f.write(lyrics)
    
    # 3) 指定 genre 提示
    genre_file = os.path.join(temp_out_dir, "genre.txt")
    with open(genre_file, "w") as f:
        f.write("pop upbeat vocal electronic") 

    # 4) 拼装命令
    #    用上面通过环境变量获取的 YUE_PYTHON & YUE_INFER_PY
    cmd = [
        YUE_PYTHON,
        YUE_INFER_PY,
        "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-cot",
        "--stage2_model", "m-a-p/YuE-s2-1B-general",
        "--genre_txt", genre_file,
        "--lyrics_txt", lyrics_file,
        "--run_n_segments", "2",
        "--stage2_batch_size", "4",
        "--output_dir", temp_out_dir,
        "--cuda_idx", "0",
        "--max_new_tokens", "3000"
    ]

    # 5) 运行子进程
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=YUE_CWD  # 指定工作目录
        )
        st.write("YuE 推理输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("YuE 推理失败，错误信息:")
        st.error(e.stderr)
        raise

    # 6) 找到生成的音频，读到内存里返回给前端
    wav_files = glob.glob(os.path.join(temp_out_dir, "*.wav")) + glob.glob(os.path.join(temp_out_dir, "*.mp3"))
    if not wav_files:
        raise FileNotFoundError(f"未在输出目录找到生成的音频文件。输出目录内容：{os.listdir(temp_out_dir)}")
    out_audio_path = wav_files[0]
    with open(out_audio_path, "rb") as f:
        audio_data = f.read()
    
    return audio_data


# =============== 7) Streamlit 主 UI ===============
col_left, col_right = st.columns([1.4, 1.6], gap="medium")

with col_left:
    st.markdown("**在这里画画**", unsafe_allow_html=True)
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

with col_right:
    st.markdown("**生成结果**", unsafe_allow_html=True)
    st.write("完成绘画后，将生成歌词并调用 YuE 直接生成完整歌曲。")

    # 生成歌词和标题
    if st.button("生成歌词"):
        if canvas_result.image_data is None:
            st.error("请先在左侧画布上绘制内容！")
        else:
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
            title = generate_song_title(image)
            raw_lyrics = generate_lyrics_with_ollama(image)
            lyrics = format_text(raw_lyrics)
            st.session_state["song_title"] = title
            st.session_state["lyrics"] = lyrics

    # 显示生成的歌词和标题
    if st.session_state["song_title"] and st.session_state["lyrics"]:
        st.markdown(f"**歌曲标题：** {st.session_state['song_title']}", unsafe_allow_html=True)
        lyrics_html = st.session_state["lyrics"].replace("\n", "<br>")
        st.markdown(f"<div class='lyrics-container'><p>{lyrics_html}</p></div>", unsafe_allow_html=True)

    # 使用 YuE 生成完整歌曲
    if st.button("生成完整歌曲"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            # 调用 YuE 推理函数，从文本直接生成完整歌曲音频
            final_audio = yue_infer(st.session_state["lyrics"])
            st.audio(final_audio, format="audio/wav")
            st.download_button("下载完整歌曲", final_audio, "full_song.wav", mime="audio/wav")
