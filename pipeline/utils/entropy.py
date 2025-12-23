import math

def semantic_entropy(clusters):
    total = sum(len(v) for v in clusters.values())
    entropy = 0.0

    for cluster in clusters.values():
        p = len(cluster) / total
        entropy -= p * math.log(p + 1e-9)

    return round(entropy, 3)
