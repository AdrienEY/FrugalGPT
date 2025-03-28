# Interaction entre Azure Production RAG et FrugalGPT

Ce document décrit les interactions entre les projets `azure-production-rag` et `FrugalGPT`, ainsi que les modifications apportées pour permettre leur intégration. Il est destiné à fournir une vue d'ensemble à toute personne n'ayant pas suivi l'évolution du projet.

---

## Contexte

- **Azure Production RAG** : Ce projet implémente un système de génération de réponses augmentées par récupération (Retrieval-Augmented Generation, RAG) en production sur Azure. Il est conçu pour intégrer des modèles de langage avancés et des pipelines de recherche documentaire.

- **FrugalGPT** : Ce projet vise à optimiser les coûts d'utilisation des modèles de langage en combinant des stratégies d'efficacité computationnelle et des modèles légers. Il repose sur des techniques avancées telles que la quantification des modèles, la distillation des connaissances et l'utilisation de modèles spécialisés pour des tâches spécifiques.

L'objectif principal de l'intégration est de tirer parti des optimisations de FrugalGPT pour réduire les coûts d'inférence dans le pipeline RAG d'Azure Production tout en maintenant une qualité de réponse élevée.

---

![Env](./image/archi.png)

## Modifications apportées

### Dans `azure-production-rag`

1. **Ajout de l'intégration avec FrugalGPT** :
   - Une nouvelle couche d'interaction a été ajoutée pour déléguer certaines tâches d'inférence à FrugalGPT.
   - Les appels aux modèles lourds ont été remplacés ou complétés par des appels aux modèles optimisés de FrugalGPT, réduisant ainsi la consommation de ressources.

2. **Mise à jour des configurations** :
   - Les fichiers de configuration (par exemple, `azure.yaml`) ont été modifiés pour inclure des paramètres spécifiques à FrugalGPT, tels que les points de terminaison API, les clés d'authentification et les stratégies d'optimisation (par exemple, seuils de latence et budgets de coût).

3. **Scripts d'inférence** :
   - Le fichier `locustfile.py` a été mis à jour pour inclure des tests de charge simulant l'utilisation conjointe des deux systèmes, en mesurant les performances et les coûts.

### Dans `FrugalGPT`

1. **Support des pipelines RAG** :
   - FrugalGPT a été étendu pour inclure des fonctionnalités spécifiques aux pipelines RAG, comme :
     - Le traitement des requêtes contextuelles pour garantir une cohérence dans les réponses.
     - La gestion des documents récupérés pour optimiser leur utilisation dans les modèles légers.

2. **API d'intégration** :
   - Une API REST a été développée pour permettre une interaction fluide avec Azure Production RAG. Cette API inclut :
     - Des points de terminaison pour l'inférence optimisée.
     - Des métriques en temps réel sur les coûts et la latence.
     - Une gestion des erreurs robuste pour garantir la fiabilité.

3. **Optimisation des modèles** :
   - Les modèles utilisés dans FrugalGPT ont été ajustés pour répondre aux besoins spécifiques du pipeline RAG :
     - **Quantification** : Réduction de la taille des modèles pour diminuer la latence et la consommation mémoire.
     - **Distillation** : Utilisation de modèles distillés pour conserver les performances tout en réduisant les coûts.
     - **Spécialisation** : Développement de modèles spécialisés pour des tâches spécifiques, comme le scoring ou la génération de réponses courtes.

4. **Stratégies d'efficacité computationnelle** :
   - Implémentation de mécanismes de fallback, où les modèles légers sont utilisés en priorité, et les modèles lourds ne sont appelés qu'en cas de besoin.
   - Introduction d'un système de gestion dynamique des ressources pour ajuster les modèles en fonction des contraintes de coût et de latence.

---

## Points d'interaction

1. **Pipeline d'inférence** :
   - Azure Production RAG délègue certaines étapes d'inférence (par exemple, la génération de réponses ou le scoring) à FrugalGPT via l'API REST.
   - FrugalGPT renvoie des résultats optimisés, qui sont ensuite intégrés dans le pipeline global.

2. **Gestion des coûts** :
   - FrugalGPT fournit des métriques détaillées sur les coûts d'inférence, comme :
     - Le coût par requête.
     - Les économies réalisées par rapport à l'utilisation exclusive de modèles lourds.
   - Ces métriques sont utilisées par Azure Production RAG pour ajuster dynamiquement les stratégies d'exécution.

3. **Partage de données** :
   - Les deux projets partagent des données telles que :
     - Les documents récupérés pour garantir une cohérence dans les réponses.
     - Les journaux d'utilisation pour affiner les modèles et les stratégies d'optimisation.

---

## Instructions pour les développeurs

1. **Configuration** :
   - Assurez-vous que les fichiers de configuration dans `azure-production-rag` incluent les paramètres nécessaires pour se connecter à FrugalGPT.
   - Vérifiez que l'API de FrugalGPT est déployée et accessible. Les paramètres requis incluent :
     - L'URL de l'API.
     - Les clés d'authentification.
     - Les seuils de latence et de coût.

2. **Tests** :
   - Utilisez les tests de charge dans `locustfile.py` pour valider les performances de l'intégration.
   - Exécutez les tests unitaires et d'intégration dans les deux projets pour garantir la stabilité.
   - Vérifiez les métriques de coût et de latence pour vous assurer que les objectifs d'optimisation sont atteints.

3. **Déploiement** :
   - Déployez les deux systèmes conjointement en suivant les instructions spécifiques à chaque projet.
   - Surveillez les performances et ajustez les configurations si nécessaire.

---

## Conclusion

Cette intégration permet de combiner la puissance des pipelines RAG d'Azure Production avec les optimisations avancées de FrugalGPT, offrant ainsi un système performant, économique et adaptable. Pour toute question ou problème, veuillez consulter les fichiers `CONTRIBUTING.md` et `CHANGELOG.md` dans les deux projets. N'hésitez pas à soumettre des suggestions ou des rapports de bogues via les dépôts GitHub respectifs.