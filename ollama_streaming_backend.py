"""
Ollama Streaming Backend for AI Storyteller
Optimized for Aya model with Lithuanian language
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
import json
import uvicorn

app = FastAPI(title="Ollama Aya Streaming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class StreamRequest(BaseModel):
    messages: List[Message]
    model: str = "aya"
    max_tokens: int = 100
    temperature: float = 0.8

@app.post("/stream")
async def stream_completion(request: StreamRequest):
    """
    Stream AI responses from Ollama Aya model
    Optimized for Lithuanian storytelling
    """
    
    async def generate_stream():
        try:
            # Prepare messages for Ollama
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Call Ollama streaming API
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    'http://localhost:11434/api/chat',
                    json={
                        "model": request.model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens,
                            "top_p": 0.9,
                            "top_k": 40
                        }
                    }
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Ollama error: {error_text.decode()}"
                        )
                    
                    # Stream chunks
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                
                                # Extract content
                                if 'message' in data and 'content' in data['message']:
                                    content = data['message']['content']
                                    if content:
                                        # Send in SSE format (Server-Sent Events)
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                
                                # Check if done
                                if data.get('done', False):
                                    yield "data: [DONE]\n\n"
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.RequestError as e:
            error_msg = f"Connection error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
        except Exception as e:
            error_msg = f"Streaming error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/generate")
async def generate_completion(request: StreamRequest):
    """
    Non-streaming endpoint (for compatibility)
    """
    try:
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                'http://localhost:11434/api/chat',
                json={
                    "model": request.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    }
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Ollama request failed"
                )
            
            data = response.json()
            text = data['message']['content']
            
            return {
                "success": True,
                "text": text,
                "model": request.model
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if Ollama is accessible"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get('http://localhost:11434/api/tags')
            
            if response.status_code == 200:
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                
                return {
                    "status": "healthy",
                    "ollama_running": True,
                    "models_available": models,
                    "aya_installed": any('aya' in m for m in models)
                }
            else:
                return {
                    "status": "unhealthy",
                    "ollama_running": False,
                    "error": "Cannot connect to Ollama"
                }
                
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_running": False,
            "error": str(e)
        }

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get('http://localhost:11434/api/tags')
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                return {
                    "models": [
                        {
                            "name": m['name'],
                            "size": m.get('size', 0),
                            "modified": m.get('modified_at', '')
                        }
                        for m in models
                    ]
                }
            else:
                raise HTTPException(status_code=500, detail="Cannot list models")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🤖 Ollama Aya Streaming Backend")
    print("="*60)
    print("\nChecking Ollama connection...")
    
    import asyncio
    
    async def check():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get('http://localhost:11434/api/tags')
                if response.status_code == 200:
                    data = response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    print(f"✅ Ollama is running")
                    print(f"✅ Available models: {', '.join(models)}")
                    
                    if any('aya' in m for m in models):
                        print(f"✅ Aya model is installed")
                    else:
                        print(f"⚠️  Aya model not found. Install with: ollama pull aya")
                else:
                    print(f"❌ Ollama not accessible")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print(f"   Make sure Ollama is running (ollama serve)")
    
    asyncio.run(check())
    
    print("\nStarting server on http://localhost:5005")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5005)