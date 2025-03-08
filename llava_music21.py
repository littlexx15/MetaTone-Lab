import subprocess
import json
import tempfile
import os
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random
import music21
import pyphen
from midi2audio import FluidSynth

from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

# 确保 set_page_config 是第一个 Streamlit 命令
st.set_page_config(page_title="MetaTone Lab", layout="wide")

import sys
import pandas as pd
# 调试输出（如果需要调试信息，建议使用 print 而不是 st.write，以避免干扰 set_page_config 调用）
print("Python executable:", sys.executable)
print("Pandas path:", pd.__file__)
print("Has DataFrame:", hasattr(pd, 'DataFrame'))

# SoundFont 路径
SOUNDFONT_PATH = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/soundfonts/VocalsPapel.sf2"

# 使用 session_state 存储当前生成的歌词和标题
if "lyrics" not in st.session_state:
    st.session_state["lyrics"] = None
if "song_title" not in st.session_state:
    st.session_state["song_title"] = None

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

st.markdown("<h1>MetaTone 实验室</h1>", unsafe_allow_html=True)

# -------------------------------
# 1️⃣ 生成歌词（调用 llava:7b）
# -------------------------------
def generate_lyrics_with_ollama(image: Image.Image) -> str:
    """
    调用 llava:7b 模型，根据图像生成英文歌词
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
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    lyrics = "".join(parsed).strip()
    return lyrics.strip('"')

# -------------------------------
# 2️⃣ 生成歌曲标题（调用 llava:7b）
# -------------------------------
def generate_song_title(image: Image.Image) -> str:
    """
    调用 llava:7b 模型，为图像生成歌曲标题
    """
    temp_path = create_temp_file(image)
    prompt = """
Provide a concise, creative, and poetic song title. Only output the title, with no extra words or disclaimers.
    """
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    title = "".join(parsed).strip()
    return title.strip('"')

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
# 4️⃣ 基于歌词生成匹配的旋律 MIDI
# -------------------------------
def split_into_syllables(line: str) -> list:
    dic = pyphen.Pyphen(lang='en')
    words = line.split()
    syllables = []
    for word in words:
        syl = dic.inserted(word)
        syllables.extend(syl.split('-'))
    return syllables

def generate_melody_for_line(line: str) -> list:
    scale_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
    syllables = split_into_syllables(line)
    melody = []
    for i, syl in enumerate(syllables):
        pitch = scale_notes[i % len(scale_notes)]
        melody.append((pitch, 1.0))
    return melody

def generate_melody_from_lyrics(lyrics: str) -> bytes:
    from music21 import stream, note, instrument
    s = stream.Stream()
    inst = instrument.Instrument()
    inst.midiProgram = 53
    s.insert(0, inst)
    lines = [l for l in lyrics.split("\n") if l.strip()]
    for line in lines:
        melody = generate_melody_for_line(line)
        for pitch, duration in melody:
            n = note.Note(pitch, quarterLength=duration)
            n.lyric = line
            s.append(n)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi_path = tmp.name
    s.write("midi", fp=midi_path)
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    os.remove(midi_path)
    return midi_bytes

def generate_matched_melody(lyrics: str) -> bytes:
    return generate_melody_from_lyrics(lyrics)

# -------------------------------
# 5️⃣ MIDI 转 WAV（基础方法）
# -------------------------------
def midi_to_wav(midi_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
        tmp_midi.write(midi_bytes)
        midi_path = tmp_midi.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    fs = FluidSynth(sound_font=SOUNDFONT_PATH)
    fs.midi_to_audio(midi_path, wav_path)
    with open(wav_path, "rb") as f:
        wav_data = f.read()
    os.remove(midi_path)
    os.remove(wav_path)
    return wav_data

# -------------------------------
# 6️⃣ So‑VITS‑SVC 推理函数
# -------------------------------
def so_vits_svc_infer(rough_wav: bytes, svc_config: str, svc_model: str) -> bytes:
    """
    将基础音频 rough_wav 输入 So‑VITS‑SVC 推理脚本，转换为更自然的英文歌声。
    svc_config: 配置文件路径 (例如 "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/configs/config.json")
    svc_model: 模型文件路径 (例如 "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/models/G_800.pth")
    """
    source_path = "source.wav"
    with open(source_path, "wb") as f:
        f.write(rough_wav)
    output_path = "converted.wav"
    cmd = [
        "python",
        "/Users/xiangxiaoxin/Documents/GitHub/so-vits-svc/inference_main.py",
        "-c", svc_config,
        "-m", svc_model,
        "-i", source_path,
        "-o", output_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, cwd="/Users/xiangxiaoxin/Documents/GitHub/so-vits-svc")
        st.write("推理输出：", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("So‑VITS‑SVC 推理失败，错误信息：")
        st.error(e.stderr)
        raise
    with open(output_path, "rb") as f:
        converted_data = f.read()
    os.remove(source_path)
    os.remove(output_path)
    return converted_data

# -------------------------------
# 7️⃣ Streamlit 主布局
# -------------------------------
col_left, col_right = st.columns([1.4, 1.6], gap="medium")

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

with col_right:
    st.markdown("<div class='subheader-text'>生成结果</div>", unsafe_allow_html=True)
    st.write("点击【生成歌词】生成歌曲标题和歌词；点击【生成基础演唱】生成基础演唱；点击【生成 So‑VITS 演唱】转换为自然的英文歌声。")
    
    if st.button("🎶 生成歌词"):
        if canvas_result.image_data is None:
            st.error("请先在左侧画布上绘制内容！")
        else:
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
            title = generate_song_title(image)
            raw_lyrics = generate_lyrics_with_ollama(image)
            lyrics = format_text(raw_lyrics)
            st.session_state["song_title"] = title
            st.session_state["lyrics"] = lyrics

    if st.session_state["song_title"] and st.session_state["lyrics"]:
        st.markdown("**歌曲标题：**", unsafe_allow_html=True)
        st.markdown(f"<div class='song-title'>{st.session_state['song_title']}</div>", unsafe_allow_html=True)
        st.markdown("**生成的歌词：**", unsafe_allow_html=True)
        lyrics_html = st.session_state["lyrics"].replace("\n", "<br>")
        st.markdown(f"<div class='lyrics-text lyrics-container'><p>{lyrics_html}</p></div>", unsafe_allow_html=True)

    if st.button("🎤 生成基础演唱"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            midi_bytes = generate_matched_melody(st.session_state["lyrics"])
            rough_wav = midi_to_wav(midi_bytes)
            st.audio(rough_wav, format="audio/wav")
            st.download_button(
                label="下载基础演唱 WAV",
                data=rough_wav,
                file_name="rough_melody.wav",
                mime="audio/wav"
            )

    if st.button("🎤 生成 So‑VITS 演唱"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            midi_bytes = generate_matched_melody(st.session_state["lyrics"])
            rough_wav = midi_to_wav(midi_bytes)
            svc_config = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/configs/config.json"
            svc_model = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/models/G_800.pth"
            converted_wav = so_vits_svc_infer(rough_wav, svc_config, svc_model)
            st.audio(converted_wav, format="audio/wav")
            st.download_button(
                label="下载 So‑VITS 演唱 WAV",
                data=converted_wav,
                file_name="converted_singing.wav",
                mime="audio/wav"
            )
