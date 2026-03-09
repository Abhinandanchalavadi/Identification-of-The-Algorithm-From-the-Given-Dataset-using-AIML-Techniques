def recommend_algorithm(performance):
    if not performance:
        return ("Insufficient Data", 0.0)
    best_algo, best_score = max(performance, key=lambda x: x[1])
    return best_algo, best_score