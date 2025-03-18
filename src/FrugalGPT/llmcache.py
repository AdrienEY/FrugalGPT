import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import logging

class LLMCache:
    def __init__(self, cache_type="tfidf", similarity_threshold=0.95):
        self.cache = {}
        self.cache_type = cache_type
        self.similarity_threshold = similarity_threshold
        
        if cache_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.query_vectors = None
            
    def find_similar_query(self, query):
        logging.info(f"Searching cache for query: {query}")
        logging.info(f"Cache size: {len(self.cache)}")
        
        if not self.cache:
            logging.info("Cache is empty")
            return None, None
            
        result = None, None
        if self.cache_type == "exact":
            result = self._exact_match(query)
        elif self.cache_type == "tfidf":
            result = self._tfidf_match(query)
        elif self.cache_type == "levenshtein":
            result = self._levenshtein_match(query)
            
        if result[0] is not None:
            logging.info(f"Cache hit! Found matching query: {result[0]}")
        else:
            logging.info("Cache miss")
        return result
            
    def _exact_match(self, query):
        if query in self.cache:
            return query, self.cache[query]
        return None, None
        
    def _tfidf_match(self, query):
        if self.query_vectors is None:
            # First time: fit vectorizer and transform all queries
            queries = list(self.cache.keys())
            self.query_vectors = self.vectorizer.fit_transform(queries)
            
        # Transform new query
        query_vec = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = (self.query_vectors @ query_vec.T).toarray().flatten()
        max_idx = similarities.argmax()
        
        if similarities[max_idx] >= self.similarity_threshold:
            cached_query = list(self.cache.keys())[max_idx]
            return cached_query, self.cache[cached_query]
        return None, None
        
    def _levenshtein_match(self, query):
        best_ratio = 0
        best_match = None
        
        for cached_query in self.cache:
            ratio = SequenceMatcher(None, query, cached_query).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cached_query
                
        if best_ratio >= self.similarity_threshold:
            return best_match, self.cache[best_match]
        return None, None

    def add_to_cache(self, query, response, model_used):
        logging.info(f"Adding to cache - Query: {query}")
        self.cache[query] = {'response': response, 'model_used': model_used}
        if self.cache_type == "tfidf":
            self.query_vectors = None  # Reset vectors to force recomputation

    def save(self, filepath):
        logging.info(f"Saving cache to: {filepath}")
        cache_data = {query: {'response': data['response'], 'model_used': data['model_used']} 
                     for query, data in self.cache.items()}
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Cache saved with {len(self.cache)} entries")

    def load(self, filepath):
        logging.info(f"Loading cache from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)
        if self.cache_type == "tfidf":
            self.query_vectors = None  # Reset vectors to force recomputation
        logging.info(f"Cache loaded with {len(self.cache)} entries")
