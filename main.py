from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from dotenv import load_dotenv
import tiktoken


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
    conversation_history: list = []  # Ajout de l'historique de la conversation si nécessaire
    system_prompt: str = ""  # Ajout du prompt système si nécessaire
    few_shots: list = []  # Ajout des exemples de few-shots si nécessaire
    content: str

# Charger les variables d'environnement
load_dotenv()

def execute_cascade_logic(prompt: str, content: str, system_prompt: str, few_shots: list):
    # Initialiser la cascade avec des paramètres par défaut
    MyCascade = FrugalGPT.LLMCascade_cache()
    strategy_path = 'strategy/cascade_strategy.json'
    
    try:
        # Paramètres fixes
        budget = 0.0005733333333333334
        MyCascade.load(loadpath=strategy_path, budget=budget)
        
        # Paramètres de génération fixes
        genparams = FrugalGPT.GenerationParameter(
            max_tokens=200,
            temperature=0.1,
            stop=['\n']
        )

        # Utiliser get_completion au lieu de get_completion_batch
        answer, model_used = MyCascade.get_completion(
            query=prompt,
            genparams=genparams,
            system_prompt = system_prompt,
            content = content,
            few_shots = few_shots
        )

        
        
        return {
            "answer": answer,
            "cost": MyCascade.get_cost(),
            "model_used": model_used
        }
            
    except Exception as e:
        raise Exception(f"Erreur lors de l'exécution de la cascade : {str(e)}")

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def execute_gpt4o(prompt: str, content: str, system_prompt: str, few_shots: list):
    try:
        # Initialiser le fournisseur de modèle GPT-4o
        provider = AzureGPT4oModelProvider("gpt-4o-mini")

        # Définir les paramètres de génération
        genparams = GenerationParameter(
            max_tokens=200,
            temperature=0.1,
            stop=['\n']
        )

        # Calcul des tokens d'entrée
        input_text = prompt + system_prompt + content + " ".join(map(str, few_shots))
        input_tokens = count_tokens(input_text)

        # Obtenir la complétion du modèle
        completion = provider.getcompletiongpt4o(
            query=prompt,
            genparams=genparams,
            content=content,
            system_prompt=system_prompt,
            few_shots=few_shots
        )

        # Calcul des tokens de sortie
        output_tokens = count_tokens(completion)

        COST_PER_INPUT_TOKENS = 0.0000025  
        COST_PER_OUTPUT_TOKENS = 0.00001

        # Calcul du coût total
        cost = (input_tokens) * COST_PER_INPUT_TOKENS + (output_tokens) * COST_PER_OUTPUT_TOKENS

        return {
            "answer": completion,
            "cost": cost,  # Arrondi pour un affichage clair
            "model_used": "gpt-4o-mini"
        }

    except Exception as e:
        raise Exception(f"Error executing GPT-4o request: {str(e)}")


@app.post("/execute_cascade")
async def execute_cascade(request: CascadeRequest):
    try:
        result = execute_cascade_logic(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            content = request.content,
            few_shots=request.few_shots
            #conversation_history=request.conversation_history
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/execute_gpt4o")
async def execute_gpt4o_endpoint(request: CascadeRequest):
    try:
        result = execute_gpt4o(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            few_shots=request.few_shots,
            content=request.content  # Vérifiez si ce paramètre est réellement utilisé
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

