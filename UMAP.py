import umap
import numpy as np

def compute(input: np.ndarray, dims: int = 2, n_of_neighbours: int = 15, num_of_iterations: int = 500) -> np.ndarray:
    model = umap.UMAP(n_components=dims, n_neighbors=n_of_neighbours, n_epochs=num_of_iterations)
    out = model.fit_transform(input)
    return out