def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    if not a or not a[0]:
        return []
    
    m, n = len(a), len(a[0])
    # Initialize an n x m matrix filled with zeros
    transposed = [[0] * m for _ in range(n)]
    
    for i in range(m):
        for j in range(n):
            transposed[j][i] = a[i][j]
            
    return transposed