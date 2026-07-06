# memory/external_memory.py
import faiss
import numpy as np

class ExternalMemory:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.values = []

    def add(self, keys, values):
        self.index.add(keys.astype(np.float32))
        self.values.extend(values)

    def search(self, query, k=5):
        D, I = self.index.search(query.astype(np.float32), k)
        return [self.values[i] for i in I[0]]
