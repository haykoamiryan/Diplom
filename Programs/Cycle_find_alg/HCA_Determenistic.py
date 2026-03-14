import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    m, n = map(int, input("Enter matrix size m n (5–1000): ").split())

    if m < 5 or m > 1000 or n < 5 or n > 1000:
        print("Matrix size must be 5–1000.")
        return

    if m % 2 == 1 and n % 2 == 1:
        print("Both odd → reducing n by 1.")
        n -= 1

    start_x = random.randint(0, m - 1)
    start_y = random.randint(0, n - 1)

    cycle = build_universal_cycle(m, n)

    start_index = cycle.index((start_x, start_y))
    cycle = cycle[start_index:] + cycle[:start_index]

    print("Hamiltonian cycle created.")
    plot_cycle(cycle, m, n)

def build_universal_cycle(rows, cols):
    transposed = False
    
    if rows % 2 != 0:
        rows, cols = cols, rows
        transposed = True
    
    cycle = []
    
    for r in range(rows):
        cycle.append((r, 0))
        
    for r in range(rows - 1, -1, -1):
        if r % 2 == 1:
            for c in range(1, cols):
                cycle.append((r, c))
        else:
            for c in range(cols - 1, 0, -1):
                cycle.append((r, c))
    
    if transposed:
        cycle = [(c, r) for r, c in cycle]
        
    return cycle

def plot_cycle(cycle, m, n):
    matrix = np.zeros((m, n))
    for i, (x, y) in enumerate(cycle):
        matrix[x, y] = i + 1 

    fig, ax = plt.subplots(figsize=(n, m))
    cax = ax.imshow(matrix, cmap='viridis', origin='upper')
    plt.colorbar(cax, label='Step number')

    for i in range(1, len(cycle)):
        x0, y0 = cycle[i-1]
        x1, y1 = cycle[i]
        if abs(x1 - x0) + abs(y1 - y0) == 1:
            ax.plot([y0, y1], [x0, x1], color='red', linewidth=2)
            ax.plot(y1, x1, 'ro', markersize=4)

    x_start, y_start = cycle[0]
    x_end, y_end = cycle[-1]
    if abs(x_start - x_end) + abs(y_start - y_end) == 1:
         ax.plot([y_end, y_start], [x_end, x_start], color='red', linewidth=2)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.set_yticks([])
    ax.set_title("Hamiltonian Cycle (Universal Logic)")
    plt.show()

if __name__ == "__main__":
    main()
