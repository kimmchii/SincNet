import numpy as np

vectors = np.load('./d_vector_timit.npy', allow_pickle=True)


vectors = vectors.item()

embeded_vectors = []
for key, value in vectors.items():
    embeded_vectors.append(value)


if __name__ == "__main__":
    print(embeded_vectors[0].shape)