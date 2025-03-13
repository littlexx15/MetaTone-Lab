import streamlit as st
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

print("Python executable:", sys.executable)

# =============== 你的辅助函数 ===============
from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

# =============== SoundFont 路径（请确保路径正确）===============
SOUNDFONT_PATH = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/soundfonts/VocalsPapel.sf2"

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
st.markdown("<h1>MetaTone 实验室</h1>", unsafe_allow_html=True)


# =============== 1) 生成歌词 (调用 llava:7b) ===============
def generate_lyrics_with_ollama(image: Image.Image) -> str:
    """
    调用 llava:7b 模型，根据图像生成双语歌词：
    每句中文歌词下方紧跟一行括号内的英文翻译，便于外国人阅读理解。
    """
    temp_path = create_temp_file(image)
    prompt = """
你是一位富有创意的歌曲写作助手。
请观察我提供的图像，根据图像内容创作一首中文歌曲。要求如下：
1. 歌词需包含不同部分，如【主歌】、【副歌】等。
2. 每一句中文歌词下方请另起一行，用括号括住对应的英文翻译，确保中文和英文分别独占一行。
3. 歌词要求充满诗意、意境深远、情感真挚。
请只输出歌词文本，不要额外说明。
    """
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    lyrics = "".join(parsed).strip()
    return lyrics.strip('"')


# =============== 2) 生成歌曲标题 (调用 llava:7b) ===============
def generate_song_title(image: Image.Image) -> str:
    """调用 llava:7b 模型，为图像生成歌曲标题（中文）。"""
    temp_path = create_temp_file(image)
    prompt = """
请为我提供一个简洁、富有诗意的中文歌曲标题，只输出标题，不要其他文字。
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


# =============== 4) 基于歌词生成匹配的旋律 MIDI（带音节到 note.lyric） ===============
def split_into_syllables(line: str) -> list:
    """将整行拆分为音节或单词。"""
    dic = pyphen.Pyphen(lang='en')
    words = line.split()
    syllables = []
    for word in words:
        syl = dic.inserted(word)
        splitted = syl.split('-')
        print(f"[DEBUG] word={word}, splitted={splitted}")
        syllables.extend(splitted)
    return syllables

def generate_melody_for_line(line: str) -> list:
    """给一行歌词生成音符，默认使用 C 大调（C4~B4），每个音节1拍。"""
    scale_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
    syllables = split_into_syllables(line)
    melody = []
    for i, syl in enumerate(syllables):
        pitch = scale_notes[i % len(scale_notes)]
        melody.append((pitch, 1.0, syl))
    return melody

def generate_melody_from_lyrics(lyrics: str, debug_save: bool = False) -> bytes:
    from music21 import stream, note, instrument
    s = stream.Stream()
    inst = instrument.Instrument()
    inst.midiProgram = 53
    s.insert(0, inst)
    lines = [l for l in lyrics.split("\n") if l.strip()]
    for line in lines:
        melody_line = generate_melody_for_line(line)
        for (pitch, dur, syl) in melody_line:
            n = note.Note(pitch, quarterLength=dur)
            n.lyric = syl
            print(f"[DEBUG] note={pitch}, lyric={repr(syl)}")
            s.append(n)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi_path = tmp.name
    s.write("midi", fp=midi_path)
    with open(midi_path, "rb") as f:
        midi_bytes = f.read()
    if debug_save:
        with open("debug_midi.mid", "wb") as debug_file:
            debug_file.write(midi_bytes)
        print("Saved debug_midi.mid")
    os.remove(midi_path)
    return midi_bytes

def generate_matched_melody(lyrics: str, debug_save: bool = False) -> bytes:
    return generate_melody_from_lyrics(lyrics, debug_save=debug_save)


# =============== 5) MIDI -> WAV（粗糙演唱） ===============
def midi_to_wav(midi_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
        midi_path = tmp_midi.name
        tmp_midi.write(midi_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    fs = FluidSynth(sound_font=SOUNDFONT_PATH)
    fs.midi_to_audio(midi_path, wav_path)
    with open(wav_path, "rb") as f:
        wav_data = f.read()
    os.remove(midi_path)
    os.remove(wav_path)
    return wav_data


# =============== 6) DiffSinger 推理函数 ===============
def diffsinger_infer(lyrics: str, config_path: str, model_path: str) -> bytes:
    """
    使用 DiffSinger 从歌词生成合成演唱。
    本函数假定你已有修改后的 ds_e2e.py 推理脚本，
    它接受 --config, --model, --lyrics, --out 参数生成 WAV 文件。
    """
    # 将歌词保存到临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        lyrics_file = tmp.name
        tmp.write(lyrics)
    
    # 创建一个临时输出文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        out_wav = tmp_wav.name

    # 构造推理命令
    cmd = [
        "/opt/anaconda3/envs/diffsinger_env/bin/python",
        "/Users/xiangxiaoxin/Documents/GitHub/DiffSinger/inference/svs/ds_e2e.py",
        "--config", "diffsinger/0228_opencpop_ds100_rel/config.yaml",
        "--model", "diffsinger/0228_opencpop_ds100_rel/model_ckpt_steps_160000.ckpt",
        "--lyrics", lyrics_file,
        "--out", out_wav
    ]
    
    try:
        # 使用 DiffSinger 根目录作为工作目录
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=True, cwd="/Users/xiangxiaoxin/Documents/GitHub/DiffSinger")
        st.write("DiffSinger 推理输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("DiffSinger 推理失败，错误信息:")
        st.error(e.stderr)
        raise

    # 读取生成的 WAV 文件
    with open(out_wav, "rb") as f:
        wav_data = f.read()

    # 清理临时文件
    os.remove(lyrics_file)
    os.remove(out_wav)
    return wav_data



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
    st.write("完成绘画后，生成中文歌词及每句对应的英文翻译，再生成基础演唱，并使用 DiffSinger 转换为自然的中文歌声（同时附英文翻译）。")
    
    # 生成歌词
    if st.button("🎶 生成双语歌词"):
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
    
    # 生成基础演唱（MIDI→WAV）
    if st.button("🎤 生成基础演唱"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            midi_bytes = generate_matched_melody(st.session_state["lyrics"], debug_save=True)
            rough_wav = midi_to_wav(midi_bytes)
            st.audio(rough_wav, format="audio/wav")
            st.download_button("下载基础演唱 WAV", rough_wav, "rough_melody.wav", mime="audio/wav")
    
    # 使用 DiffSinger 生成合成演唱
    if st.button("🎤 生成 DiffSinger 演唱"):
        if not st.session_state["lyrics"]:
            st.error("请先生成歌词！")
        else:
            # 请将下面的 config_path 与 model_path 修改为你自己的 DiffSinger 配置文件与模型路径
            diffsinger_config = "/path/to/diffsinger/config.json"
            diffsinger_model = "/path/to/diffsinger/model.pth"
            synthesized_wav = diffsinger_infer(st.session_state["lyrics"], diffsinger_config, diffsinger_model)
            st.audio(synthesized_wav, format="audio/wav")
            st.download_button("下载 DiffSinger 演唱 WAV", synthesized_wav, "diffsinger_singing.wav", mime="audio/wav")
