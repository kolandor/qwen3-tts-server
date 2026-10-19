import io
import threading

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

app = FastAPI(title="Local Qwen3 TTS")

print("Loading Qwen3-TTS on GPU...")

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

lock = threading.Lock()

class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(min_length=1, max_length=1500)
    voice: str = "Serena"
    response_format: str = "wav"
    speed: float = 1.0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/voices")
def voices():
    names = tts.get_supported_speakers() or []
    return {"voices": [
        {"id": name, "name": name}
        for name in names
    ]}

@app.post("/v1/audio/speech")
def generate_speech(request: SpeechRequest):
    if request.response_format.lower() != "wav":
        raise HTTPException(
            status_code=400,
            detail="Only WAV is supported"
        )

    voice = request.voice or "Serena"
    if voice.lower() in ("default", "alloy"):
        voice = "Serena"

    try:
        with lock:
            wavs, sr = tts.generate_custom_voice(
                text=request.input,
                language="Russian",
                speaker=voice,
            )
    except ValueError as error:
        raise HTTPException(
            status_code=400, detail=str(error)
        )

    audio = io.BytesIO()
    sf.write(
        audio, wavs[0], sr,
        format="WAV", subtype="PCM_16"
    )
    return Response(
        content=audio.getvalue(),
        media_type="audio/wav"
    )
