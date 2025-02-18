@ -64,13 +64,12 @@ def extract_facial_features(image_path):
# -------------------------------
# 3️⃣ 生成歌词（Ollama / Gemma:2b）
# -------------------------------
def generate_lyrics(facial_features):
    """确保歌词长度足够"""
    prompt = f"A poetic song (at least 15 words) about a person with {facial_features['face_shape']} face and {facial_features['skin_color']} skin."
def generate_lyrics(facial_features, emotion):
    """结合面部特征和情绪生成歌词"""
    prompt = f"A poetic song about a person with {facial_features['face_shape']} face and {facial_features['skin_color']} skin, feeling {emotion}."
    
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    
    # 确保歌词足够长
    lyrics = response['message']['content']
    if len(lyrics.split()) < 15:
        lyrics += " This song is full of emotions and melodies that flow smoothly."
@ -78,6 +77,7 @@ def generate_lyrics(facial_features):
    return lyrics



# -------------------------------
# 4️⃣ 生成旋律（PyTorch 版音乐生成）
# -------------------------------
@ -105,17 +105,16 @@ def generate_melody(emotion):


# -------------------------------
# 5️⃣ 使用 Speedy-Speech 进行歌曲合成
# 5️⃣ 使用 FastPitch 进行歌曲合成
# -------------------------------
def synthesize_song(lyrics, melody_path):
    """使用 Speedy-Speech 进行歌唱合成"""
    """使用 FastPitch 进行语音合成"""
    
    # 加载 Speedy-Speech
    tts = TTS("tts_models/en/ljspeech/speedy-speech")  # ✅ 无需 espeak-ng

    # 生成歌唱语音
    tts = TTS("tts_models/en/ljspeech/fast_pitch")  # ✅ 改用 FastPitch，速度更快
    output_wav = "output.wav"
    tts.tts_to_file(text=lyrics, file_path=output_wav)
    
    # 生成语音并加快语速，防止声音拉长
    tts.tts_to_file(text=lyrics, file_path=output_wav, speed=1.1, max_decoder_steps=500)

    return output_wav

@ -127,14 +126,26 @@ def synthesize_song(lyrics, melody_path):
def process_image(image):
    """完整的 AI 音乐生成流程"""
    cv2.imwrite("input.jpg", image)
      
    # 检测情绪
    emotion = detect_emotion("input.jpg")
    print(f"🧐 识别的情绪：{emotion}")  # ✅ 打印情绪识别结果

    # 提取面部特征
    features = extract_facial_features("input.jpg")
    lyrics = generate_lyrics(features)
    
    # 生成歌词（结合面部特征 & 情绪）
    lyrics = generate_lyrics(features, emotion)
    
    # 生成旋律（基于情绪）
    melody = generate_melody(emotion)
    
    # 合成歌曲
    song = synthesize_song(lyrics, melody)
    
    return lyrics, melody, song


interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
