import random
import math

def generate_matrix(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def matrix_multiply(A, B):
    cols_B = len(B[0])
    rows_A = len(A)
    cols_A = len(A[0])
    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return result

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def normalize(vec):
    magnitude = math.sqrt(sum(x**2 for x in vec))
    return [x / magnitude for x in vec] if magnitude != 0 else vec

def flatten(matrix):
    return [val for row in matrix for val in row]

def softmax(logits):
    exps = [math.exp(x - max(logits)) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cosine_similarity(a, b):
    dot = dot_product(a, b)
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0

def batch_normalize(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean)**2 for x in data) / len(data)
    std = math.sqrt(variance + 1e-8)
    return [(x - mean) / std for x in data]

if __name__ == "__main__":
    M = generate_matrix(4, 4)
    flat = flatten(M)
    normed = normalize(flat)
    probs = softmax(normed[:10])
    print("Softmax output:", [round(p, 4) for p in probs])
