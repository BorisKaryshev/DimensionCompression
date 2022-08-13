import numpy as np

def compute(input: np.ndarray, dims: int = 2, pereplexity: float = 15, number_of_iterations: int = 500) -> np.ndarray:
    from sklearn.manifold import TSNE 
    embed = TSNE(
        n_components=dims, 
        perplexity=pereplexity, 
        early_exaggeration=10,     
        learning_rate=200, 
        n_iter=number_of_iterations, 
        n_iter_without_progress=300, 
        min_grad_norm=0.0000001, 
        metric='euclidean', 
        init='random',
        verbose=0, 
        random_state=42, 
        method='barnes_hut', 
        angle=0.5, 
        n_jobs=-1, 
    )

    output = embed.fit_transform(input)

    print('Kullback-Leibler divergence after optimization: ', embed.kl_divergence_)
    print('No. of iterations: ', embed.n_iter_)

    return output