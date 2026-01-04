"""
Ultravox Inference Server
Handles audio input and returns text transcription
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import numpy as np
import io
import uvicorn

app = FastAPI(title="Ultravox Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
processor = None
device = None

@app.on_event("startup")
async def load_model():
    """Load Ultravox model on startup"""
    global model, processor, device
    
    print("Loading speech recognition model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use Whisper for now (more stable and widely compatible)
    # Ultravox requires specific setup - will add later
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        print("Loading Whisper model (supports Lithuanian)...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print(f"✅ Whisper model loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), language: str = "lt"):
    """
    Transcribe audio file to text
    
    Args:
        audio: Audio file (wav, mp3, etc.)
        language: Target language code (default: 'lt' for Lithuanian)
    """
    try:
        # Read audio file
        audio_bytes = await audio.read()
        audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Process audio
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=448,
                language=language if hasattr(model.config, 'language') else None
            )
        
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return {
            "transcription": transcription,
            "language": language,
            "sample_rate": sample_rate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/transcribe-realtime")
async def transcribe_realtime(audio: UploadFile = File(...)):
    """Real-time transcription for streaming audio"""
    # Implement streaming transcription logic
    return {"status": "streaming not implemented yet"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

@app.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported languages"""
    # Ultravox v0.6 supports 42 languages including Lithuanian
    languages = [
        "ar", "be", "bn", "bg", "zh", "cs", "da", "nl", "en", "et",
        "fi", "fr", "gl", "ka", "de", "el", "hi", "hu", "it", "ja",
        "lv", "lt", "mk", "mr", "fa", "pl", "pt", "ro", "ru", "sr",
        "sk", "sl", "es", "sw", "sv", "ta", "th", "tr", "uk", "ur", "vi"
    ]
    return {
        "languages": languages,
        "count": len(languages)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)