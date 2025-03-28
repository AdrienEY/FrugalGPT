from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from dotenv import load_dotenv
import tiktoken
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime

sys.path.insert(0, 'src/')
import FrugalGPT
import service
from service.modelservice import AzureGPT4oModelProvider, GenerationParameter

# Configure logging
logging.basicConfig(level=logging.INFO)

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
    logging.info("Starting cascade logic execution")
    MyCascade = FrugalGPT.LLMCascade_cache()
    strategy_path = 'strategy/cascade_strategy.json'
    
    try:
        budget = 0.0005733333333333334
        logging.info(f"Loading cascade strategy from: {strategy_path} with budget: {budget}")
        MyCascade.load(loadpath=strategy_path, budget=budget)
        
        genparams = FrugalGPT.GenerationParameter(
            max_tokens=200,
            temperature=0.1,
            stop=['\n']
        )
        logging.info(f"Generation parameters: {genparams}")

        answer, model_used = MyCascade.get_completion(
            query=prompt,
            genparams=genparams,
            system_prompt=system_prompt,
            content=content,
            few_shots=few_shots
        )
        logging.info(f"Cascade answer: {answer}, Model used: {model_used}")

        cost = MyCascade.get_cost()
        logging.info(f"Cascade cost: {cost}")

        return {
            "answer": answer,
            "cost": cost,
            "model_used": model_used
        }
    except Exception as e:
        logging.error(f"Error in cascade logic: {str(e)}")
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

@app.post("/compare_costs")
async def compare_costs(request: CascadeRequest):
    try:
        # Log the incoming request
        logging.info("Received request for /compare_costs")

        # Execute cascade logic
        cascade_result = execute_cascade_logic(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            content=request.content,
            few_shots=request.few_shots
        )
        cascade_cost = cascade_result["cost"]
        logging.info(f"Cascade cost calculated: {cascade_cost}")

        # Execute GPT-4o logic
        gpt4o_result = execute_gpt4o(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            content=request.content,
            few_shots=request.few_shots
        )
        gpt4o_cost = gpt4o_result["cost"]
        logging.info(f"GPT-4o cost calculated: {gpt4o_cost}")

        # Create the comparison graph
        labels = ['Cascade', 'GPT-4o']
        costs = [cascade_cost, gpt4o_cost]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, costs, color=['blue', 'orange'])
        plt.title('Cost Comparison')
        plt.ylabel('Cost (USD)')
        plt.xlabel('Model')
        plt.tight_layout()

        # Annotate the bars with the cost values
        for bar, cost in zip(bars, costs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position
                bar.get_height(),  # Y position
                f"${cost:.6f}",  # Text to display
                ha='center', va='bottom'  # Center alignment
            )

        # Ensure the 'plots' directory exists
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Generate a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"cost_comparison_{timestamp}.png")

        # Save the graph to the file
        plt.savefig(plot_path)
        logging.info(f"Graph saved to file: {plot_path}")

        # Encode the graph as base64 for the JSON response
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        logging.info("Graph generated and encoded successfully")

        # Return the JSON response
        return JSONResponse(content={
            "status": "success",
            "graph": graph_base64,
            "details": {
                "cascade_cost": cascade_cost,
                "gpt4o_cost": gpt4o_cost,
                "graph_path": plot_path  # Include the file path in the response
            }
        })
    except Exception as e:
        logging.error(f"Error in /compare_costs: {str(e)}")
        return {"status": "error", "message": str(e)}

