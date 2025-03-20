import streamlit as st
st.set_page_config(page_title="MetaTone Lab", layout="wide")

import os
import subprocess
import tempfile
import glob
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Environment variables for YuE inference
YUE_PYTHON   = os.environ.get("YUE_PYTHON",   "python")
YUE_INFER_PY = os.environ.get("YUE_INFER_PY", "yueForWindows/inference/infer.py")
YUE_CWD      = os.environ.get("YUE_CWD",      "yueForWindows")

from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

# Store lyrics and title in session state
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

# 1) Generate lyrics using LLava:7B
def generate_lyrics_with_ollama(image: Image.Image) -> str:
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

# 2) Generate song title using LLava:7B
def generate_song_title(image: Image.Image) -> str:
    temp_path = create_temp_file(image)
    prompt = """
Provide a concise, creative, and poetic song title. Only output the title, with no extra words or disclaimers.
    """
    stream = analyze_image_file(image_file=temp_path, model="llava:7b", user_prompt=prompt)
    parsed = stream_parser(stream)
    title = "".join(parsed).strip()
    return title.strip('"')

# 3) Format lyrics for display
def format_text(text: str) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [l[0].upper() + l[1:] if l else "" for l in lines]
    return "\n\n".join(lines)

# 4) Run YuE inference to generate music from text
def yue_infer(lyrics: str) -> bytes:
    # Create temp output directory
    temp_out_dir = tempfile.mkdtemp()
    
    # Write lyrics
    lyrics_file = os.path.join(temp_out_dir, "lyrics.txt")
    with open(lyrics_file, "w") as f:
        f.write(lyrics)
    
    # Specify simple genre prompt
    genre_file = os.path.join(temp_out_dir, "genre.txt")
    with open(genre_file, "w") as f:
        f.write("pop upbeat vocal electronic") 

    # Build command
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

    # Run YuE inference
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=YUE_CWD
        )
        st.write("YuE inference output:", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("YuE inference failed:")
        st.error(e.stderr)
        raise

    # Find and return generated audio
    wav_files = glob.glob(os.path.join(temp_out_dir, "*.wav")) + glob.glob(os.path.join(temp_out_dir, "*.mp3"))
    if not wav_files:
        raise FileNotFoundError(f"No audio found in {temp_out_dir}. Contents: {os.listdir(temp_out_dir)}")
    out_audio_path = wav_files[0]
    with open(out_audio_path, "rb") as f:
        audio_data = f.read()
    return audio_data

# 5) Streamlit UI
col_left, col_right = st.columns([1.4, 1.6], gap="medium")

with col_left:
    st.markdown("**Draw here**", unsafe_allow_html=True)
    st.write("Select color/size and sketch on the canvas.")
    brush_color = st.color_picker("Brush color", value="#000000")
    brush_size = st.slider("Brush size", 1, 50, value=5)
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
    st.markdown("**Generate results**", unsafe_allow_html=True)
    st.write("After drawing, generate lyrics and have YuE synthesize a full song.")

    # Generate lyrics and song title
    if st.button("Generate lyrics"):
        if canvas_result.image_data is None:
            st.error("Please sketch something first!")
        else:
            image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert("RGB")
            title = generate_song_title(image)
            raw_lyrics = generate_lyrics_with_ollama(image)
            lyrics = format_text(raw_lyrics)
            st.session_state["song_title"] = title
            st.session_state["lyrics"] = lyrics

    # Display title and lyrics
    if st.session_state["song_title"] and st.session_state["lyrics"]:
        st.markdown(f"**Song Title:** {st.session_state['song_title']}", unsafe_allow_html=True)
        lyrics_html = st.session_state["lyrics"].replace("\n", "<br>")
        st.markdown(f"<div class='lyrics-container'><p>{lyrics_html}</p></div>", unsafe_allow_html=True)

    # Generate full song from lyrics
    if st.button("Generate complete song"):
        if not st.session_state["lyrics"]:
            st.error("Please generate lyrics first!")
        else:
            final_audio = yue_infer(st.session_state["lyrics"])
            st.audio(final_audio, format="audio/wav")
            st.download_button("Download Full Song", final_audio, "full_song.wav", mime="audio/wav")
