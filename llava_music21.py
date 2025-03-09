import streamlit as st
# 必须最先调用 set_page_config
st.set_page_config(page_title="MetaTone Lab", layout="wide")

import sys
import os
import subprocess
import tempfile
import json
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random
import music21
import pyphen
from midi2audio import FluidSynth
import torch

# 调试输出
print("Python executable:", sys.executable)

# 导入自定义辅助函数（确保 util 文件夹在项目根目录下）
from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

# SoundFont 路径（请确保路径正确）
SOUNDFONT_PATH = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/soundfonts/VocalsPapel.sf2"

# 使用 session_state 存储歌词和标题
if "lyrics" not in st.session_state:
    st.session_state["lyrics"] = None
if "song_title" not in st.session_state:
    st.session_state["song_title"] = None

# 页面样式（仅调用一次）
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
    """, unsafe_allow_html=True
)
st.markdown("<h1>MetaTone 实验室</h1>", unsafe_allow_html=True)


############################################
# 1️⃣ 生成歌词（调用 llava:7b）
############################################
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
    stream = analyze_image_file(
        image_file=temp_path,
        model="llava:7b",
        user_prompt=prompt
    )
    parsed = stream_parser(stream)
    lyrics = "".join(parsed).strip()
    return lyrics.strip('"')

############################################
# 2️⃣ 生成歌曲标题（调用 llava:7b）
############################################
def generate_song_title(image: Image.Image) -> str:
    """调用 llava:7b 模型，为图像生成歌曲标题。"""
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
    return title.strip('"')

############################################
# 3️⃣ 格式化歌词
############################################
def format_text(text: str) -> str:
    """去除多余空行，并保证每行首字母大写。"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [l[0].upper() + l[1:] if l else "" for l in lines]
    return "\n\n".join(lines)

############################################
# 4️⃣ 基于歌词生成匹配的旋律 MIDI
############################################
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
    """对外暴露的函数，从歌词生成对应的 MIDI 文件并返回其二进制内容。"""
    return generate_melody_from_lyrics(lyrics)

############################################
# 5️⃣ MIDI 转 WAV（粗糙演唱）
############################################
def midi_to_wav(midi_bytes: bytes) -> bytes:
    """将 MIDI 二进制内容转换成 WAV 音频（粗糙演唱）。"""
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

############################################
# 6️⃣ So‑VITS‑SVC 推理函数
############################################
def so_vits_svc_infer(rough_wav: bytes, svc_config: str, svc_model: str) -> bytes:
    """
    将基础音频 rough_wav 输入 So‑VITS‑SVC 推理脚本，转换为更自然的英文歌声。
    svc_config: 配置文件路径，例如 "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/configs/config.json"
    svc_model: 模型文件路径，例如 "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/models/G_800.pth"
    注意：请确保配置文件中 'spk' 字段包含你要使用的说话人名称，如 "hal-9000"。
    """
    svc_repo = "/Users/xiangxiaoxin/Documents/GitHub/so-vits-svc"
    raw_dir = os.path.join(svc_repo, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_name = "temp_infer.wav"
    raw_path = os.path.join(svc_repo, "raw", raw_name)
    with open(raw_path, "wb") as f:
        f.write(rough_wav)
    # 调试：保存一份粗糙音频到项目根目录，便于检查
    with open("debug_rough.wav", "wb") as f:
        f.write(rough_wav)
    cmd = [
        "python",
        os.path.join(svc_repo, "inference_main.py"),
        "-m", svc_model,
        "-c", svc_config,
        "-n", "temp_infer",   # 注意：此处不加扩展名，应与 so-vits-svc 预期一致
        "-t", "0",
        "-s", "hal-9000"      # 此处必须与配置文件中的 spk 对应
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, cwd=svc_repo)
        st.write("So‑VITS‑SVC 推理输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("So‑VITS‑SVC 推理失败，错误信息:")
        st.error(e.stderr)
        raise
    # 注意：根据你的 so-vits-svc 版本，输出文件名可能为 "temp_infer_0key_hal-9000_sovits_pm.flac"
    out_file = os.path.join(svc_repo, "results", "temp_infer_0key_hal-9000_sovits_pm.flac")
    if not os.path.exists(out_file):
        raise FileNotFoundError(f"无法找到输出文件：{out_file}\n结果文件夹内容: {os.listdir(os.path.join(svc_repo, 'results'))}")
    with open(out_file, "rb") as f:
        converted_data = f.read()
    return converted_data

############################################
# 7️⃣ Streamlit 主 UI 布局
############################################
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
    st.write("完成绘画后，可生成歌词、基础演唱，再用 So‑VITS‑SVC 转换为自然的英文歌声。")
    
    # 按钮：生成歌词
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

    # 显示生成的歌词和标题
    if st.session_state["song_title"] and st.session_state["lyrics"]:
        st.markdown(f"**歌曲标题：** {st.session_state['song_title']}", unsafe_allow_html=True)
        lyrics_html = st.session_state["lyrics"].replace("\n", "<br>")
        st.markdown(f"<div class='lyrics-container'><p>{lyrics_html}</p></div>", unsafe_allow_html=True)

    # 按钮：生成基础演唱（MIDI→WAV）
    if st.button("🎤 生成基础演唱"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            midi_bytes = generate_matched_melody(st.session_state["lyrics"])
            rough_wav = midi_to_wav(midi_bytes)
            st.audio(rough_wav, format="audio/wav")
            st.download_button("下载基础演唱 WAV", rough_wav, "rough_melody.wav", mime="audio/wav")

    # 按钮：使用 So‑VITS‑SVC 生成自然演唱
    if st.button("🎤 生成 So‑VITS‑SVC 演唱"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            midi_bytes = generate_matched_melody(st.session_state["lyrics"])
            rough_wav = midi_to_wav(midi_bytes)
            svc_config = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/configs/config.json"
            svc_model = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/models/G_800.pth"
            converted_wav = so_vits_svc_infer(rough_wav, svc_config, svc_model)
            st.audio(converted_wav, format="audio/wav")
            st.download_button("下载 So‑VITS‑SVC 演唱 WAV", converted_wav, "converted_singing.flac", mime="audio/flac")
