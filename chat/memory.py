import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChatMemory:
    """Simple in-memory vector store using TF-IDF for retrieval."""

    def __init__(self):
        self.texts: list[str] = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None

    def add(self, text: str) -> None:
        """Add a piece of text to memory."""
        self.texts.append(text)
        # Re-fit vectorizer with all texts for simplicity
        self.vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve top_k most relevant memories for the query."""
        if not self.texts:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.texts[i] for i in top_indices if sims[i] > 0]