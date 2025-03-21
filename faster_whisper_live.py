import queue
import numpy as np
import torch
import pyaudio
import time
import faster_whisper

# 모델 설정
model_size = "large-v2"  # 사용할 모델 크기
model = faster_whisper.WhisperModel(model_size, device="cpu")

# 오디오 스트리밍 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 샘플링 속도 (16kHz)
CHUNK = 1024  # 버퍼 크기
RECORD_SECONDS = 3  # 정확히 3초씩 오디오를 수집

audio_queue = queue.Queue()
processing = False  # STT 변환 중 여부 플래그

def audio_callback(in_data, frame_count, time_info, status):
    """마이크에서 받은 오디오 데이터를 큐에 저장 (STT 변환 중이 아닐 때만)"""
    if not processing:
        audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# PyAudio 스트리밍 시작
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

stream.start_stream()
print("🎤 Real-time STT started. Speak into the microphone...")

try:
    segment_index = 0  # 음성 인식 구간 인덱스 (3초 단위 증가)

    while True:
        audio_buffer = b""
        start_time = time.time()

        # 🎯 **정확히 3초 동안만 오디오 수집**
        while time.time() - start_time < RECORD_SECONDS:
            while not audio_queue.empty():
                audio_buffer += audio_queue.get()

        if len(audio_buffer) > 0:
            # 🛑 STT 변환 중에는 입력 차단
            processing = True
            stream.stop_stream()

            # 오디오 데이터를 NumPy 배열로 변환
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # Faster Whisper 실행
            segments, info = model.transcribe(audio_np, beam_size=5)

            # 📌 **출력 시간을 정확히 3초 단위로 유지**
            start_segment = segment_index * RECORD_SECONDS
            end_segment = (segment_index + 1) * RECORD_SECONDS

            for segment in segments:
                print(f"[{start_segment:.2f}s - {end_segment:.2f}s] {segment.text}")

            # 다음 3초 블록으로 이동
            segment_index += 1

            # 🔄 변환 완료 후 다시 오디오 입력 받기
            processing = False
            stream.start_stream()

except KeyboardInterrupt:
    print("🛑 Stopping real-time STT...")
    stream.stop_stream()
    stream.close()
    p.terminate()
