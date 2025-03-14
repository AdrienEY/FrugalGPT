from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from cascade.strategy import CascadeStrategy
from cascade.models import get_completion

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A modifier en production pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CascadeRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    strategy: Dict[str, Any]
    budget: float

def execute_cascade_logic(prompt: str, max_tokens: int, temperature: float, strategy: Dict[str, Any], budget: float):
    cascade_strategy = CascadeStrategy(strategy)
    generation_params = {
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    result = cascade_strategy.execute(
        prompt=prompt,
        budget=budget,
        generation_params=generation_params,
        get_completion_func=get_completion
    )
    return result

@app.post("/execute_cascade")
async def execute_cascade(request: CascadeRequest):
    try:
        result = execute_cascade_logic(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            strategy=request.strategy,
            budget=request.budget
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
