from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path
import hashlib

class LLMCache:
    def __init__(self, cache_file="cache/query_cache.json", similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        self.cache_file = Path(cache_file)
        path = str(Path(__file__).parent.parent.parent.parent / "all-mpnet-base-v2")
        self.model = SentenceTransformer(path)

        # Charger le cache existant si disponible
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                try:
                    self.cache = json.load(f)
                except json.JSONDecodeError:
                    self.cache = []
        else:
            self.cache = []

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def add_to_cache(self, query, response, model_used):
        query_embedding = self.model.encode([query])[0].tolist()

        entry = {
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "model": model_used
        }

        self.cache.append(entry)
        self.save_cache()

    def get_from_cache(self, query):
        if not self.cache:
            return None, None

        query_embedding = self.model.encode([query])[0]

        for entry in self.cache:
            cached_embedding = entry["embedding"]
            similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]

            if similarity >= self.similarity_threshold:
                return entry["response"], entry["model"]

        return None, None