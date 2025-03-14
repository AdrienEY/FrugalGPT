from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from dotenv import load_dotenv
sys.path.insert(0, 'src/')
import FrugalGPT

app = FastAPI()


# uvicorn main:app --reload

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CascadeRequest(BaseModel):
    prompt: str

# Charger les variables d'environnement
load_dotenv()

def execute_cascade_logic(prompt: str):
    # Initialiser la cascade avec des paramètres par défaut
    MyCascade = FrugalGPT.LLMCascade()
    strategy_path = 'strategy/cascade_strategy.json'
    
    try:
        # Paramètres fixes
        budget = 0.0005733333333333334
        MyCascade.load(loadpath=strategy_path, budget=budget)
        
        # Paramètres de génération fixes
        genparams = FrugalGPT.GenerationParameter(
            max_tokens=50,
            temperature=0.1,
            stop=['\n']
        )
        
        # Utiliser get_completion au lieu de get_completion_batch
        answer, model_used = MyCascade.get_completion(
            query=prompt,
            genparams=genparams
        )
        
        return {
            "answer": answer,
            "cost": MyCascade.get_cost(),
            "model_used": model_used
        }
            
    except Exception as e:
        raise Exception(f"Erreur lors de l'exécution de la cascade : {str(e)}")

@app.post("/execute_cascade")
async def execute_cascade(request: CascadeRequest):
    try:
        result = execute_cascade_logic(
            prompt=request.prompt
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
