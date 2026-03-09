import os
import time
from collections import deque

def build_neighbors(m: int, n: int) -> list[list[int]]:
    neighbors = []
    for r in range(m):
        for c in range(n):
            nb = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < n:
                    nb.append(rr * n + cc)
            neighbors.append(nb)
    return neighbors

def is_connected(unvisited_mask: int, start: int, neighbors: list[list[int]]) -> bool:
    if unvisited_mask == 0:
        return True
    seen = 1 << start
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nb in neighbors[cell]:
            bit = 1 << nb
            if (unvisited_mask & bit) and not (seen & bit):
                seen |= bit
                queue.append(nb)
    return (seen & unvisited_mask) == unvisited_mask

def count_hamiltonian_cycles(m: int, n: int) -> int:
    total = m * n
    if total < 3:
        return 0
    if total % 2 == 1:
        return 0

    neighbors = build_neighbors(m, n)
    start = 0
    full_mask = (1 << total) - 1

    count = 0
    init_visited = 1 << start
    stack: list[tuple[int, int, int, int]] = [(start, init_visited, 0, 1)]

    start_time = time.time()

    while stack:
        cell, visited, nb_idx, path_len = stack[-1]

        if nb_idx >= len(neighbors[cell]):
            stack.pop()
            continue

        stack[-1] = (cell, visited, nb_idx + 1, path_len)

        nb = neighbors[cell][nb_idx]

        if path_len == total:
            if nb == start:
                count += 1
                if count % 2 == 0:
                    elapsed = time.time() - start_time
                    print(f"\rFound: {count // 2:,} | Time: {elapsed:.2f}s", end="", flush=True)
            continue

        if visited & (1 << nb):
            continue

        new_visited = visited | (1 << nb)
        remaining = full_mask & ~new_visited

        check_mask = remaining | (1 << start)
        if check_mask and not is_connected(check_mask, nb, neighbors):
            continue

        stack.append((nb, new_visited, 0, path_len + 1))

    return count // 2

def format_time_readable(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    if m > 0:
        return f"{m}m {int(s)}s"
    else:
        return f"{s:.2f}s"

def get_table_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Calculae table.md")

def find_next_easiest_task(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Table file not found at {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tasks = []

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("| **") and "** |" in stripped:
            parts = stripped.split("|")
            try:
                row_label = parts[1].strip().replace("*", "")
                m = int(row_label)
            except ValueError:
                continue

            for n in range(1, 31):
                col_idx = n + 1
                if col_idx < len(parts):
                    cell_content = parts[col_idx].strip()
                    if cell_content == "":
                        tasks.append((m, n))
    
    if not tasks:
        return None

    tasks.sort(key=lambda x: (x[0] * x[1], x[0]))
    return tasks[0]

def update_table_file(file_path, m, n, result, time_str):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    table_updated = False

    for line in lines:
        if not table_updated and line.strip().startswith(f"| **{m}** |"):
            parts = line.split("|")
            col_idx = n + 1
            if col_idx < len(parts):
                parts[col_idx] = f" {result:,} "
                new_line = "|".join(parts)
                new_lines.append(new_line)
                table_updated = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    has_log_header = False
    has_table_header = False
    
    for line in new_lines:
        if "### Execution Logs" in line:
            has_log_header = True
        if "| Grid Size |" in line and "| Cycles Found |" in line:
            has_table_header = True

    if not has_log_header:
        if new_lines and not new_lines[-1].strip() == "":
            new_lines.append("\n")
        new_lines.append("\n### Execution Logs\n\n")

    if not has_table_header:
        new_lines.append("| Grid Size | Cycles Found | Time |\n")
        new_lines.append("| :---: | :---: | :---: |\n")

    log_entry = f"| {m} x {n} | {result:,} | {time_str} |\n"
    new_lines.append(log_entry)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    table_path = get_table_path()
    print(f"Tracking table file: {table_path}")

    while True:
        task = find_next_easiest_task(table_path)
        if not task:
            print("\nAll cells in the table are filled!")
            break

        m, n = task
        print(f"\n[Auto-Bot] Processing grid {m}x{n} (complexity: {m*n})...")

        start_time = time.time()
        result = count_hamiltonian_cycles(m, n)
        end_time = time.time()

        elapsed = end_time - start_time
        time_str = format_time_readable(elapsed)

        print(f"\n[Done] {m}x{n} = {result:,} (Time: {time_str})")
        
        print("Updating table...", end=" ")
        update_table_file(table_path, m, n, result, time_str)
        print("Saved.")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
