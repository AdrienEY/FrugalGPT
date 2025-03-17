from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from dotenv import load_dotenv

sys.path.insert(0, 'src/')
import FrugalGPT
import service
from service.modelservice import AzureGPT4oModelProvider, GenerationParameter

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

def execute_gpt4o(prompt: str):
    try:
        # Initialize the GPT4o provider
        provider = AzureGPT4oModelProvider("gpt-4o")
        
        # Set generation parameters
        genparams = GenerationParameter(
            max_tokens=50,
            temperature=0.1,
            stop=['\n']
        )
        
        # Get completion from the model
        result = provider.getcompletion(
            context=prompt,
            genparams=genparams
        )
        
        return {
            "answer": result["completion"],
            "cost": provider._get_cost(prompt, result),
            "model_used": "gpt-4o"
        }
            
    except Exception as e:
        raise Exception(f"Error executing GPT4o request: {str(e)}")

@app.post("/execute_cascade")
async def execute_cascade(request: CascadeRequest):
    try:
        result = execute_cascade_logic(
            prompt=request.prompt
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/execute_gpt4o")
async def execute_gpt4o_endpoint(request: CascadeRequest):
    try:
        result = execute_gpt4o(
            prompt=request.prompt
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
