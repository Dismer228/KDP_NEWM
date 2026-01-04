"""
AI Storyteller Backend
Handles LLM API calls for continuous storytelling
Supports: OpenAI, Anthropic Claude, Local Ollama
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn

# Optional imports - install what you need
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

app = FastAPI(title="AI Storyteller Backend")

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

class StoryRequest(BaseModel):
    model: str  # 'openai', 'claude', 'ollama'
    api_key: Optional[str] = None
    messages: List[Message]
    max_tokens: int = 150
    temperature: float = 0.8

@app.post("/generate")
async def generate_story_part(request: StoryRequest):
    """
    Generate next part of the story using specified LLM
    """
    try:
        if request.model == 'openai':
            if not HAS_OPENAI:
                raise HTTPException(status_code=500, detail="OpenAI library not installed")
            return await generate_openai(request)
        
        elif request.model == 'claude':
            if not HAS_ANTHROPIC:
                raise HTTPException(status_code=500, detail="Anthropic library not installed")
            return await generate_claude(request)
        
        elif request.model == 'ollama':
            return await generate_ollama(request)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_openai(request: StoryRequest):
    """Generate using OpenAI GPT"""
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key required for OpenAI")
    
    openai.api_key = request.api_key
    
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    text = response.choices[0].message.content
    
    return {
        "success": True,
        "text": text,
        "model": "openai-gpt-4"
    }

async def generate_claude(request: StoryRequest):
    """Generate using Anthropic Claude"""
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key required for Claude")
    
    client = anthropic.Anthropic(api_key=request.api_key)
    
    # Separate system message from conversation
    system_message = next((m.content for m in request.messages if m.role == 'system'), '')
    conversation = [{"role": m.role, "content": m.content} 
                   for m in request.messages if m.role != 'system']
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=request.max_tokens,
        system=system_message,
        messages=conversation
    )
    
    text = response.content[0].text
    
    return {
        "success": True,
        "text": text,
        "model": "claude-3-sonnet"
    }

async def generate_ollama(request: StoryRequest):
    """Generate using local Ollama"""
    import httpx
    
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:11434/api/chat',
            json={
                "model": "llama2",
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            },
            timeout=30.0
        )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Ollama request failed")
    
    data = response.json()
    text = data['message']['content']
    
    return {
        "success": True,
        "text": text,
        "model": "ollama-llama2"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": {
            "openai": HAS_OPENAI,
            "claude": HAS_ANTHROPIC,
            "ollama": True  # Always available if Ollama server running
        }
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models = []
    
    if HAS_OPENAI:
        models.append({
            "id": "openai",
            "name": "OpenAI GPT-4",
            "requires_api_key": True
        })
    
    if HAS_ANTHROPIC:
        models.append({
            "id": "claude",
            "name": "Anthropic Claude 3",
            "requires_api_key": True
        })
    
    models.append({
        "id": "ollama",
        "name": "Local Ollama (Llama 2)",
        "requires_api_key": False
    })
    
    return {"models": models}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🎭 AI Storyteller Backend")
    print("="*60)
    print("\nAvailable models:")
    print(f"  OpenAI: {'✅' if HAS_OPENAI else '❌ (pip install openai)'}")
    print(f"  Claude: {'✅' if HAS_ANTHROPIC else '❌ (pip install anthropic)'}")
    print(f"  Ollama: ✅ (requires Ollama server running)")
    print("\nStarting server on http://localhost:5005")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5004)
