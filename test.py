import numpy as np
import lib

def compute_score(input: np.ndarray, y: np.ndarray):
    score = int(0)
    for i in range(len(input)):
        if(np.argmax(input[i]) == y[i]):
            score += 1

    return np.float128(score/len(input))

def helper(x: np.ndarray, y: np.ndarray, num_of_iterations: int = 3):
    out = []
    out.append(lib.grad_boost_test(x, y))
    out.append(lib.knn_test(x, y))
    out.append(lib.kmean_test(x, y))
    num_of_iterations -= 1
    for i in range(num_of_iterations):
        out[0] += lib.grad_boost_test(x, y)
        out[1] += lib.knn_test(x, y)
        out[2] += lib.kmean_test(x, y)

    out[0] /= (num_of_iterations+1)
    out[1] /= (num_of_iterations+1)
    out[2] /= (num_of_iterations+1)

    return np.array(out)

def add_to_table(table, base_score: np.array, projection_score: np.array, n_of_parametrs:int = 0, name: str="", dataset_name: str = "", add_inf: str = ""):
    x = [name]
    x.append(dataset_name)
    x.append(n_of_parametrs)
    for i in range(len(base_score)):
        x.append(base_score[i])
        x.append(projection_score[i])
    x.append(add_inf)
    table.append(x)
    return table

def test_dataset(dataset: np.ndarray, dataset_classification: np.ndarray, table, dataset_name, stat_iter: int = 3, depth:int = 3, gr_boost_iter: int = 10):
    print("Dataset", dataset_name, " proccesing started")
    n_of_params = len(dataset[0])
    main_score = np.float128(0)

    for i in range(stat_iter):
        main_score += helper(dataset, dataset_classification)
    main_score /= np.float128(stat_iter)

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.AE_compute(dataset, dims=2, num_of_iterations=100)
        av_score += helper(tmp, dataset_classification)

    av_score /= np.float128(stat_iter)
    add_to_table(table, main_score, av_score, n_of_parametrs=n_of_params, name="AE", dataset_name=dataset_name)
    print("\tAE finished")

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.VAE_compute(dataset, num_of_iterations=100, dims=2)
        av_score += helper(tmp, dataset_classification)
    av_score /= np.float128(stat_iter)
    add_to_table(table, main_score, av_score, n_of_parametrs=n_of_params, name="VAE", dataset_name=dataset_name)
    print("\tVAE finished")

    for i in range(1, 51, 5):    
        av_score = np.float128(0)
        for j in range(stat_iter):
            tmp = lib.tSNE_compute(dataset, pereplexity=i, dims=2, number_of_iterations=500)
            av_score += helper(tmp, dataset_classification)
        av_score /= np.float128(stat_iter)
        add_to_table(table, main_score, av_score, n_of_parametrs=n_of_params, name="tSNE", dataset_name=dataset_name, add_inf="pereplexity=" + str(i))
        print("\ttSNE finished")

    for i in range(2, 51, 5):
        av_score = np.float128(0)
        for j in range(stat_iter):
            tmp = lib.umap_compute(dataset, n_of_neighbours=i, num_of_iterations=100)
            av_score += helper(tmp, dataset_classification)
        av_score /= np.float128(stat_iter)
        add_to_table(table, main_score, av_score, n_of_parametrs=n_of_params, name="UMAP", dataset_name=dataset_name, add_inf="n_of_neighbours=" + str(i))
        print("\tUMAP finished")

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.pca_compute(dataset, num_of_iterations=1000)
        av_score += helper(tmp, dataset_classification)
    av_score /= np.float128(stat_iter)
    add_to_table(table, main_score, av_score, n_of_parametrs=n_of_params, name="PCA", dataset_name=dataset_name)
    print("\tPCA finished")

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.kernelPCA_compute(dataset, num_of_iterations=1000)
        av_score += helper(tmp, dataset_classification)
    av_score /= np.float128(stat_iter)
    add_to_table(table, main_score, av_score, n_of_parametrs=n_of_params, name="KernelPCA", dataset_name=dataset_name)
    print("\tKernelPCA finished")

    print("Dataset", dataset_name, "is done")
    return table

