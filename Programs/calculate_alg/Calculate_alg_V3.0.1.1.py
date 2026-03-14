import os
import time
import sys
from numba import njit, objmode, uint64
import numpy as np

def python_print_progress(m, n, c):
    print(f"\r    Progress: Grid {m}x{n} | Current cycles: {c:,}", end="", flush=True)

@njit(cache=True)
def build_neighbors(m, n):
    total = m * n
    neighbors = np.full((total, 4), -1, dtype=np.int32)
    for r in range(m):
        for c in range(n):
            cell_idx = r * n + c
            count = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < n:
                    neighbors[cell_idx, count] = rr * n + cc
                    count += 1
    return neighbors

@njit(cache=True)
def is_connected_128(mask0, mask1, start, neighbors):
    if mask0 == 0 and mask1 == 0:
        return True
    
    seen0 = uint64(0)
    seen1 = uint64(0)
    
    if start < 64:
        seen0 |= uint64(1) << uint64(start)
    else:
        seen1 |= uint64(1) << uint64(start - 64)
    
    total_cells = neighbors.shape[0]
    queue = np.empty(total_cells, dtype=np.int32)
    head = 0
    tail = 0
    queue[tail] = start
    tail += 1
    
    while head < tail:
        cell = queue[head]
        head += 1
        for i in range(4):
            nb = neighbors[cell, i]
            if nb == -1:
                break
            
            if nb < 64:
                bit = uint64(1) << uint64(nb)
                if (mask0 & bit) and not (seen0 & bit):
                    seen0 |= bit
                    queue[tail] = nb
                    tail += 1
            else:
                bit = uint64(1) << uint64(nb - 64)
                if (mask1 & bit) and not (seen1 & bit):
                    seen1 |= bit
                    queue[tail] = nb
                    tail += 1
                    
    return (seen0 == mask0) and (seen1 == mask1)

@njit(cache=True)
def count_hamiltonian_cycles_numba_128(m, n, neighbors):
    total = m * n
    if total < 3 or total % 2 == 1:
        return 0
    start = 0
    
    full0 = uint64(0xFFFFFFFFFFFFFFFF) if total >= 64 else (uint64(1) << uint64(total)) - uint64(1)
    full1 = (uint64(1) << uint64(total - 64)) - uint64(1) if total > 64 else uint64(0)

    count = 0
    stack = np.empty((total + 1, 5), dtype=np.int64)
    stack_ptr = 0
    stack[stack_ptr] = [start, 1, 0, 0, 1]
    
    last_print_time = 0.0
    with objmode(last_print_time='float64'):
        last_print_time = time.time()
    
    iters = 0
    while stack_ptr >= 0:
        iters += 1
        if iters % 2000000 == 0:
            current_time = 0.0
            with objmode(current_time='float64'):
                current_time = time.time()
            if current_time - last_print_time >= 10.0:
                with objmode():
                    python_print_progress(m, n, count // 2)
                last_print_time = current_time

        cell, vis0, vis1, nb_idx, path_len = stack[stack_ptr]
        
        if nb_idx >= 4 or neighbors[cell, nb_idx] == -1:
            stack_ptr -= 1
            continue
            
        stack[stack_ptr, 3] += 1
        nb = neighbors[cell, nb_idx]
        
        if path_len == total:
            if nb == start:
                count += 1
            continue
            
        is_visited = False
        if nb < 64:
            is_visited = (vis0 & (uint64(1) << uint64(nb))) != 0
        else:
            is_visited = (vis1 & (uint64(1) << uint64(nb - 64))) != 0
        
        if is_visited:
            continue
        
        new_vis0, new_vis1 = uint64(vis0), uint64(vis1)
        if nb < 64:
            new_vis0 |= uint64(1) << uint64(nb)
        else:
            new_vis1 |= uint64(1) << uint64(nb - 64)
        
        rem0, rem1 = full0 & ~new_vis0, full1 & ~new_vis1
        chk0, chk1 = rem0 | uint64(1), rem1
        if not is_connected_128(chk0, chk1, start, neighbors):
            continue
            
        stack_ptr += 1
        stack[stack_ptr, 0] = nb
        stack[stack_ptr, 1] = new_vis0
        stack[stack_ptr, 2] = new_vis1
        stack[stack_ptr, 3] = 0
        stack[stack_ptr, 4] = path_len + 1
        
    return count // 2

def format_time_readable(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {int(s)}s" if m > 0 else f"{s:.2f}s"

def find_next_task(file_path):
    if not os.path.exists(file_path):
        return None
    tasks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if not line.strip().startswith("| **"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        try:
            m = int(parts[1].replace("*", ""))
        except:
            continue
        for n_idx, cell in enumerate(parts[2:-1], 1):
            if cell == "":
                tasks.append((m, n_idx))
    if not tasks:
        return None
    tasks.sort(key=lambda x: (x[0] * x[1], x[0]))
    return tasks[0]

def update_results_table(file_path, m, n, result, time_str):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines, log_idx = [], -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            parts = line.split("|")
            if len(parts) > 1:
                row_label = parts[1].strip()
                # Update (m, n)
                if row_label == f"**{m}**":
                    if n + 1 < len(parts):
                        parts[n + 1] = f" {result:,} "
                        line = "|".join(parts)
                # Update (n, m) if n != m
                elif m != n and row_label == f"**{n}**":
                    if m + 1 < len(parts):
                        parts[m + 1] = f" {result:,} "
                        line = "|".join(parts)
        
        if "### Execution Logs" in line:
            log_idx = i
        new_lines.append(line)
    
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"| {m} x {n} | {result:,} | {time_str} | {ts} |\n"
    if log_idx != -1:
        found = False
        for j in range(log_idx, min(log_idx + 10, len(new_lines))):
            if "| Grid Size |" in new_lines[j]:
                found = True
                new_lines.insert(j + 2, log_entry)
                break
        if not found:
            new_lines.insert(log_idx + 1, "\n| Grid Size | Cycles Found | Time | Finished At |\n| :---: | :---: | :---: | :---: |\n")
            new_lines.insert(log_idx + 3, log_entry)
    else:
        new_lines.append("\n### Execution Logs\n\n| Grid Size | Cycles Found | Time | Finished At |\n| :---: | :---: | :---: | :---: |\n")
        new_lines.append(log_entry)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    table_filename = "Calculate Table V3.0.1.1.md"
    table_arg = sys.argv[1] if len(sys.argv) > 1 else table_filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [os.path.join(script_dir, table_arg), table_arg, os.path.abspath(table_arg)]
    table_path = None
    for p in possible_paths:
        if os.path.exists(p):
            table_path = os.path.abspath(p)
            break
    if not table_path:
        print(f"Error: Database file '{table_arg}' not found.")
        return

    print("System initialized. Method: Hamiltonian Path Enumeration (128-bit architecture).")
    print(f"Target data file: {table_path}")

    while True:
        task = find_next_task(table_path)
        if not task:
            print("Computational tasks completed.")
            break

        m, n = task
        print(f"\nProcessing grid configuration: {m}x{n}...", end=" ", flush=True)
        neighbors = build_neighbors(m, n)
        t0 = time.time()
        result = count_hamiltonian_cycles_numba_128(m, n, neighbors)
        dt = time.time() - t0
        t_str = format_time_readable(dt)
        print(f"Completed. Result: {result:,} cycles. Elapsed time: {t_str}")
        update_results_table(table_path, m, n, result, t_str)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
