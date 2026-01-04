"""
Azure TTS Server with Streaming Support
Supports both streaming and legacy batch modes
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import azure.cognitiveservices.speech as speechsdk
import os
import base64
import asyncio
import uvicorn

app = FastAPI(title="Azure Lithuanian TTS - Streaming Enabled")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice: str = "lt-LT-OnaNeural"
    rate: str = "0%"
    pitch: str = "0%"

@app.post("/synthesize-stream")
async def synthesize_stream(request: TTSRequest):
    """
    NEW: Stream audio chunks as they're generated
    Client receives audio progressively - much faster!
    """
    try:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        service_region = os.getenv("AZURE_SPEECH_REGION", "westeurope")
        
        if not speech_key:
            raise HTTPException(status_code=500, detail="Azure key not configured")
        
        print(f"🎤 Streaming TTS: {request.text[:50]}...")
        
        async def generate_audio_stream():
            # Configure speech
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key,
                region=service_region
            )
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )
            
            # Use pull stream for streaming output
            pull_stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
            
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Create SSML
            ssml = f"""<speak version='1.0' xml:lang='lt-LT'>
                <voice name='{request.voice}'>
                    <prosody rate='{request.rate}' pitch='{request.pitch}'>
                        {request.text}
                    </prosody>
                </voice>
            </speak>"""
            
            # Start synthesis in background
            result_future = synthesizer.speak_ssml_async(ssml)
            
            # Stream chunks as they become available
            chunk_size = 4096
            total_bytes = 0
            
            while True:
                audio_chunk = pull_stream.read(chunk_size)
                if len(audio_chunk) == 0:
                    break
                    
                total_bytes += len(audio_chunk)
                yield audio_chunk
                await asyncio.sleep(0)  # Allow other coroutines to run
            
            # Wait for synthesis to complete
            result = result_future.get()
            
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"❌ Synthesis failed: {result.reason}")
            else:
                print(f"✅ Streamed {total_bytes} bytes")
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "Transfer-Encoding": "chunked"
            }
        )
        
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.post("/synthesize")
async def synthesize_legacy(request: TTSRequest):
    """
    LEGACY: Keep this for backward compatibility
    Returns complete audio as base64
    """
    try:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        service_region = os.getenv("AZURE_SPEECH_REGION", "westeurope")
        
        if not speech_key:
            raise HTTPException(status_code=500, detail="Azure key not configured")
        
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=service_region
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
        
        ssml = f"""<speak version='1.0' xml:lang='lt-LT'>
            <voice name='{request.voice}'>
                <prosody rate='{request.rate}' pitch='{request.pitch}'>
                    {request.text}
                </prosody>
            </voice>
        </speak>"""
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
            return {
                "success": True,
                "audio_base64": audio_base64,
                "format": "mp3",
                "sample_rate": 16000,
                "voice": request.voice
            }
        else:
            cancellation = result.cancellation_details
            error_msg = f"Synthesis canceled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.get("/voices")
async def get_available_voices():
    """Get available Lithuanian voices"""
    return {
        "voices": [
            {
                "name": "lt-LT-LeonasNeural",
                "gender": "Male",
                "language": "Lithuanian"
            },
            {
                "name": "lt-LT-OnaNeural",
                "gender": "Female",
                "language": "Lithuanian"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check"""
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    return {
        "status": "healthy",
        "azure_configured": bool(speech_key),
        "streaming_enabled": True
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🎤 Azure Lithuanian TTS - STREAMING MODE")
    print("="*60)
    
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "westeurope")
    
    if key:
        masked = key[:8] + "..." + key[-4:]
        print(f"✅ API Key: {masked}")
        print(f"✅ Region: {region}")
        print(f"✅ Streaming: ENABLED")
    else:
        print("❌ AZURE_SPEECH_KEY not set!")
    
    print("\nEndpoints:")
    print("  POST /synthesize-stream  - NEW streaming endpoint")
    print("  POST /synthesize         - Legacy batch endpoint")
    print("\nStarting server on http://localhost:5003")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5003)