# Interaction entre Azure Production RAG et FrugalGPT

Ce document décrit les interactions entre les projets `azure-production-rag` et `FrugalGPT`, ainsi que les modifications apportées pour permettre leur intégration. Il vise à fournir une vue d'ensemble claire et détaillée.

---

## Contexte

- **Azure Production RAG** : Ce projet implémente un système de génération de réponses augmentées par récupération (Retrieval-Augmented Generation, RAG) en production sur Azure. Il intègre des modèles de langage avancés et des pipelines de recherche documentaire.

- **FrugalGPT** : Ce projet optimise les coûts d'utilisation des modèles de langage en combinant des stratégies d'efficacité computationnelle et des modèles légers. Il repose sur des techniques telles que l'adaptation des prompts, l'approximation des modèles LLM et les cascades de modèles.

L'objectif principal de l'intégration est de tirer parti des optimisations de FrugalGPT pour réduire les coûts d'inférence dans le pipeline RAG d'Azure Production tout en maintenant une qualité de réponse élevée.

---

## Architecture du projet

![Architecture](./image/archi.png)

---

## Modifications apportées

### Dans `azure-production-rag`

1. **Intégration avec FrugalGPT** :
   - Modification du fichier `chatreadretrieveread.py` (dans le dossier `approaches` du backend) pour appeler FrugalGPT. Ce fichier transmet les informations nécessaires pour générer une réponse et obtenir une estimation des coûts associés à la requête.
   - Création du fichier `server_client.py`, appelé par `chatreadretrieveread.py`, pour gérer la communication avec FrugalGPT via le port 8000 en localhost. Ce fichier effectue :
     - Un appel à la cascade de modèles.
     - Une comparaison des coûts entre différents modèles.
     - Une réponse HTTP 200 en cas de bon fonctionnement.

### Dans `FrugalGPT`

1. **Support des pipelines RAG** :
   - Ajout de fonctionnalités spécifiques aux pipelines RAG, notamment :
     - Le traitement des requêtes contextuelles pour garantir la cohérence des réponses.
     - La gestion optimisée des documents récupérés pour les modèles légers.

2. **API d'intégration** :
   - Développement d'une API REST pour une interaction fluide avec Azure Production RAG. Cette API inclut :
     - Des points de terminaison pour l'inférence optimisée des coûts.
     - Des métriques en temps réel sur les coûts.
     - Une gestion robuste des erreurs pour garantir la fiabilité.

3. **Stratégies d'efficacité computationnelle** :
   - Implémentation de mécanismes de fallback : les modèles légers sont utilisés en priorité, et les modèles lourds ne sont appelés qu'en cas de besoin.
   - Gestion dynamique des ressources pour ajuster les modèles en fonction des contraintes de coût et de latence.
   - Mise en place d'une solution de cache dans le fichier `llmcache.py` pour réutiliser les réponses à des questions similaires. Cela évite des appels API inutiles.
   - Utilisation de la librairie `sentence-transformers` avec le modèle `all-mpnet-base-v2` disponible sur Hugging Face : [lien](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). Ce modèle peut être téléchargé localement ou appelé directement.

---

## Points d'interaction

1. **Pipeline d'inférence** :
   - Azure Production RAG délègue certaines étapes d'inférence (génération de réponses, scoring, etc.) à FrugalGPT via l'API REST.
   - FrugalGPT renvoie des résultats optimisés, intégrés ensuite dans le pipeline global.

2. **Gestion des coûts** :
   - FrugalGPT fournit des métriques détaillées, telles que :
     - Le coût par requête.
     - Les économies réalisées par rapport à l'utilisation exclusive de modèles lourds.
   - Ces métriques permettent à Azure Production RAG d'ajuster dynamiquement ses stratégies d'exécution.

3. **Partage de données** :
   - Les deux projets partagent des données, notamment :
     - Les documents récupérés pour garantir la cohérence des réponses.
     - Les journaux d'utilisation pour affiner les modèles et les stratégies d'optimisation.

---

## Instructions pour les développeurs

### Configuration dans `azure-production-rag`

1. Configurez votre environnement Azure avec votre abonnement et les services nécessaires.
2. Lancez le backend et le frontend avec la commande :
   ```bash
   ./start.ps1
   ```
3. Vérifiez que le fichier `server_client.py` (dans le dossier `core` du backend) contient les paramètres nécessaires pour établir la connexion avec FrugalGPT.

### Configuration dans `FrugalGPT`

1. Téléchargez la base de données requise avec la commande :
   ```bash
   wget -P db/ https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.sqlite
   ```
2. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```
3. Créez un fichier `.env` contenant les points de terminaison et les clés API nécessaires.

### Exécution de `FrugalGPT`

1. Testez les modèles LLM et configurez la stratégie de cascade avec :
   ```bash
   python approachfrugalgpt.py
   ```
2. Démarrez le serveur pour activer l'API REST :
   ```bash
   uvicorn main:app --reload
   ```
3. Vous pouvez exécuter les deux commandes successives avec :
   ```bash
   ./run_scripts.ps1
   ```

### Exécution conjointe de `FrugalGPT` et `azure-production-rag`

1. Lancez les commandes respectives :
   - Pour FrugalGPT : `uvicorn main:app --reload`
   - Pour Azure Production RAG : `./start.ps1`
2. Une fois les deux projets démarrés, l'interaction entre les deux repositories permet de répondre aux questions posées via l'interface frontend du Legal Copilot.

---

## Conclusion

Cette intégration combine la puissance des pipelines RAG d'Azure Production avec les optimisations avancées de FrugalGPT, offrant un système performant, économique et adaptable. Pour toute question ou problème, consultez les fichiers `CONTRIBUTING.md` et `CHANGELOG.md` dans les deux projets. N'hésitez pas à soumettre des suggestions ou des rapports de bogues via les dépôts GitHub respectifs.