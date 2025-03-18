import os
import streamlit as st
import sys
import subprocess
import tempfile
import glob
import shutil
import json
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random

# Set page configuration
st.set_page_config(page_title="MetaTone Lab", layout="wide")

print("Python executable:", sys.executable)

# Import your helper functions
from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

# SoundFont path (ensure this is correct)
SOUNDFONT_PATH = "/Users/xiangxiaoxin/Documents/GitHub/FaceTune/soundfonts/VocalsPapel.sf2"

# Initialize session state for lyrics and song title
if "lyrics" not in st.session_state:
    st.session_state["lyrics"] = None
if "song_title" not in st.session_state:
    st.session_state["song_title"] = None

# Page styling
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

# ---------- Functions ----------

def generate_lyrics_with_ollama(image: Image.Image) -> str:
    """Call llava:7b model to generate English lyrics based on the provided image."""
    temp_path = create_temp_file(image)
    prompt = """
    You are a creative songwriting assistant. 
    Please write a poetic song with a clear structure that can be sung.
    Include the following sections (all in English, in brackets):
    [VERSE] - 4 to 8 short, rhythmic lines.
    [CHORUS] - 4 to 8 short, catchy lines.
    Optionally include [BRIDGE] or [OUTRO] sections.
    Separate each section with two newlines ("\n\n"). 
    Only output the lyrics with these labels.
    Now, here is the image:
    """
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    lyrics = "".join(parsed).strip()
    return lyrics.strip('"')

def generate_song_title(image: Image.Image) -> str:
    """Call llava:7b model to generate a song title based on the provided image."""
    temp_path = create_temp_file(image)
    prompt = """
Provide a concise, creative, and poetic song title. Only output the title, with no extra words or disclaimers.
    """
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    title = "".join(parsed).strip()
    return title.strip('"')

def format_text(text: str) -> str:
    """Remove extra blank lines and ensure each line starts with uppercase."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [l[0].upper() + l[1:] if l else "" for l in lines]
    return "\n\n".join(lines)

def yue_infer(lyrics: str) -> bytes:
    """
    Run the YuE inference script with the provided lyrics.
    Subprocess output is not printed to the UI.
    """
    # Paths to the inference script and Python executable
    YUE_INFER_SCRIPT = r"C:\Users\24007516\GitHub\YuEGP\inference\infer.py"
    PYTHON_EXECUTABLE = r"C:\ProgramData\anaconda3\envs\yue4\python.exe"
    
    # Create a temporary output directory
    temp_out_dir = tempfile.mkdtemp(prefix="yue_output_")
    
    # Write lyrics to a temporary file
    lyrics_file = os.path.join(temp_out_dir, "lyrics.txt")
    with open(lyrics_file, "w", encoding="utf-8") as f:
        f.write(lyrics)
    
    # Create a simple genre file (modify as needed)
    genre_file = os.path.join(temp_out_dir, "genre.txt")
    with open(genre_file, "w", encoding="utf-8") as f:
        f.write("pop upbeat vocal electronic")
    
    # Construct the command line arguments
    cmd = [
        PYTHON_EXECUTABLE,
        YUE_INFER_SCRIPT,
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
    
    # Set CUDA environment variables
    env = os.environ.copy()
    env["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin;" + env["PATH"]
    
    # Set the working directory to the folder where infer.py lives:
    cwd = os.path.dirname(YUE_INFER_SCRIPT)
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=cwd,  # Run in the same directory as infer.py
            env=env
        )
    except subprocess.CalledProcessError as e:
        st.error("YuE inference failed. Please check your configuration and try again.")
        st.error("Subprocess error output:")
        st.error(e.stderr)
        raise
    
    # Look for output audio files (.wav or .mp3)
    wav_files = glob.glob(os.path.join(temp_out_dir, "*.wav")) + glob.glob(os.path.join(temp_out_dir, "*.mp3"))
    if not wav_files:
        raise FileNotFoundError(f"No audio file found in the output directory. Contents: {os.listdir(temp_out_dir)}")
    
    out_audio_path = wav_files[0]
    with open(out_audio_path, "rb") as f:
        audio_data = f.read()
    
    # Optionally, remove the temporary directory afterward:
    # shutil.rmtree(temp_out_dir)
    
    return audio_data

# ---------- Streamlit UI ----------

col_left, col_right = st.columns([1.4, 1.6], gap="medium")

with col_left:
    st.markdown("**Draw Your Creation Here**", unsafe_allow_html=True)
    st.write("Choose brush color and size, then draw on the canvas.")
    brush_color = st.color_picker("Brush Color", value="#000000")
    brush_size = st.slider("Brush Size", 1, 50, value=5)
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
    st.markdown("**Result**", unsafe_allow_html=True)
    st.write("After drawing, generate lyrics and then produce a full track using YuE.")
    
    if st.button("Generate Lyrics"):
        if canvas_result.image_data is None:
            st.error("Please draw something on the left canvas first!")
        else:
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
            title = generate_song_title(image)
            raw_lyrics = generate_lyrics_with_ollama(image)
            lyrics = format_text(raw_lyrics)
            st.session_state["song_title"] = title
            st.session_state["lyrics"] = lyrics

    if st.session_state["song_title"] and st.session_state["lyrics"]:
        st.markdown(f"<div class='song-title'>{st.session_state['song_title']}</div>", unsafe_allow_html=True)
        lyrics_html = st.session_state["lyrics"].replace("\n", "<br>")
        st.markdown(f"<div class='lyrics-container'><p>{lyrics_html}</p></div>", unsafe_allow_html=True)
    
    if st.button("Generate Full Song"):
        if not st.session_state["lyrics"]:
            st.error("Please generate lyrics first!")
        else:
            with st.spinner("Generating full song..."):
                try:
                    final_audio = yue_infer(st.session_state["lyrics"])
                except Exception as e:
                    st.error("Error generating full song. Please check the logs.")
                    raise
            st.audio(final_audio, format="audio/wav")
            st.download_button("Download Full Song", final_audio, "full_song.wav", mime="audio/wav")
