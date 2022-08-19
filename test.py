import numpy as np
import lib

def format_data(input: np.ndarray, y: np.ndarray):
    formated_data = np.ndarray(shape=(len(input),))
    for i in range(len(input)):
        formated_data[i] = input[i][y[i]]
    return formated_data

def add_to_table(table, score: np.float128 = 0, base_score: np.float128 = 0, name: str="", dataset_name: str = "", add_inf: str = ""):
    x = [name]
    x.append(dataset_name)
    x.append(base_score)
    x.append(score)
    x.append((score/base_score)*100)
    x.append(add_inf)
    table.append(x)
    return table

def test_dataset(dataset: np.ndarray, dataset_classification: np.ndarray, table, dataset_name, stat_iter: int = 3, depth:int = 3, gr_boost_iter: int = 10):
    print("Dataset", dataset_name, " proccesing started")
    main_score = np.float128(0)

    for i in range(stat_iter):
        probs, t= lib.grad_boost_test(dataset, dataset_classification, max_depth=depth, num_of_iterations=gr_boost_iter)
        main_score += t
    main_score /= np.float128(stat_iter)

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.AE_compute(dataset, dims=2, num_of_iterations=100)
        probs, score = lib.grad_boost_test(tmp, dataset_classification, max_depth=2, num_of_iterations=10)
        av_score += score
    av_score /= np.float128(stat_iter)
    add_to_table(table, score=av_score, base_score=main_score, name="AE", dataset_name=dataset_name)
    print("\tAE finished")

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.VAE_compute(dataset, num_of_iterations=100, dims=2)
        probs, score = lib.grad_boost_test(tmp, dataset_classification, max_depth=2, num_of_iterations=10)
        av_score += score
    av_score /= np.float128(stat_iter)
    add_to_table(table, score=av_score, base_score=main_score, name="VAE", dataset_name=dataset_name)
    print("\tVAE finished")

    for i in range(1, 51, 5):    
        av_score = np.float128(0)
        for j in range(stat_iter):
            tmp = lib.tSNE_compute(dataset, pereplexity=i, dims=2, number_of_iterations=500)
            probs, score = lib.grad_boost_test(tmp, dataset_classification, max_depth=2, num_of_iterations=10)
            av_score += score
        av_score /= np.float128(stat_iter)
        add_to_table(table, score=av_score, base_score=main_score, name="tSNE", dataset_name=dataset_name, add_inf="pereplexity = " + str(i))
        print("\ttSNE finished")

    for i in range(2, 51, 5):
        av_score = np.float128(0)
        for j in range(stat_iter):
            tmp = lib.umap_compute(dataset, n_of_neighbours=i, num_of_iterations=100)
            probs, score = lib.grad_boost_test(tmp, dataset_classification, max_depth=2, num_of_iterations=10)
            av_score += score
        av_score /= np.float128(stat_iter)
        add_to_table(table, score=av_score, base_score=main_score, name="UMAP", dataset_name=dataset_name, add_inf="n_of_neighbours = " + str(i))
        print("\tUMAP finished")

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.pca_compute(dataset, num_of_iterations=1000)
        probs, score = lib.grad_boost_test(tmp, dataset_classification, max_depth=2, num_of_iterations=10)
        av_score += score
    av_score /= np.float128(stat_iter)
    add_to_table(table, score=av_score, base_score=main_score, name="PCA", dataset_name=dataset_name)
    print("\tPCA finished")

    av_score = np.float128(0)
    for i in range(stat_iter):
        tmp = lib.kernelPCA_compute(dataset, num_of_iterations=1000)
        probs, score = lib.grad_boost_test(tmp, dataset_classification, max_depth=2, num_of_iterations=10)
        av_score += score
    av_score /= np.float128(stat_iter)
    add_to_table(table, score=av_score, base_score=main_score, name="KernelPCA", dataset_name=dataset_name)
    print("\tKernelPCA finished")

    print("Dataset", dataset_name, "is done")
    return table