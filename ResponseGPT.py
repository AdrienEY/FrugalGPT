# pip install azure-ai-inference
import os
from pathlib import Path
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, dotenv_values
import pandas as pd
from IPython.display import display
import numpy
from tqdm import tqdm
import shutil
import copy

# Add the parent directory of 'src' to Python path
#current_dir = Path(__file__).parent
#backend_dir = current_dir.parent
#src_path = os.path.join(backend_dir, 'src')
#if src_path not in sys.path:
#    sys.path.append(str(src_path))

sys.path.insert(0, 'app/backend/src/')
import FrugalGPT

load_dotenv()

#print(os.environ)



def generate_response(queries, strategy_name, budget, genparams):
    """
    Génère des réponses en utilisant FrugalGPT avec une stratégie et un budget donné.

    :param queries: Liste des requêtes utilisateur.
    :param strategy_name: Nom de la stratégie enregistrée.
    :param budget: Budget à appliquer pour la génération.
    :param genparams: Paramètres de génération (ex: max_tokens, temperature, etc.).
    :return: Les réponses générées.
    """
    # Charger la stratégie FrugalGPT
    MyCascade = FrugalGPT.LLMCascade()
    strategy_path = 'app\\backend\strategy\cascade_strategy.json'
    
    try:
        MyCascade.load(loadpath=strategy_path, budget=budget)
        print(f"Stratégie '{strategy_name}' chargée avec succès pour un budget de {budget}.")
    except Exception as e:
        print(f"Erreur lors du chargement de la stratégie : {e}")
        return None

    # Générer des réponses
    response = MyCascade.get_completion_batch(queries=[[q, "", i] for i, q in enumerate(queries)], genparams=genparams)

    if not response.empty:
        results = response.to_dict(orient='records')
        return [f"ID: {res['_id']}, Answer: {res['answer']}, Reference Answer: {res['ref_answer']}, Cost: {res['cost']}, Model Used: {res['model_used']}" for res in results]
    else:
        return ["Aucune réponse générée."]

# Exemple d'utilisation
queries = [
    "Quelle est la capitale de la France ?",
    "Qui a écrit le livre '1984' ?",
    "Quel est le plus grand océan du monde ?",
    "Combien de continents y a-t-il sur Terre ?",
    "Quelle est la planète la plus proche du Soleil ?",
    "En quelle année a eu lieu la Révolution française ?",
    "Quel est le symbole chimique de l'or ?",
    "Qui a peint la Joconde ?",
    "Quel est l'animal terrestre le plus rapide ?",
    "Quelle est la langue la plus parlée dans le monde ?"
]
strategy_name = "AGNEWS_Model20252602"
budget = 0.000090
genparams = FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])

responses = generate_response(queries, strategy_name, budget, genparams)
for response in responses:
    print("Réponse générée :", response)
