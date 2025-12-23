from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cluster_embeddings(embeddings, similarity_threshold=0.85):
    clusters = {}
    cluster_id = 0
    used = set()

    sims = cosine_similarity(embeddings)

    for i in range(len(embeddings)):
        if i in used:
            continue

        clusters[cluster_id] = [i]
        used.add(i)

        for j in range(i + 1, len(embeddings)):
            if j not in used and sims[i][j] >= similarity_threshold:
                clusters[cluster_id].append(j)
                used.add(j)

        cluster_id += 1

    return clusters
