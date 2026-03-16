import random
import time
import sys
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
matplotlib.use("TkAgg")          
sys.setrecursionlimit(10000)        
DX = (1, -1, 0, 0)
DY = (0, 0, 1, -1)
def flat(x: int, y: int, n: int) -> int:
    return x * n + y
def coords(f: int, n: int) -> tuple[int, int]:
    return divmod(f, n)
def is_valid(x: int, y: int, m: int, n: int) -> bool:
    return 0 <= x < m and 0 <= y < n
def count_free_neighbors(f: int, visited: list[bool], m: int, n: int) -> int:
    x, y = coords(f, n)
    cnt = 0
    for i in range(4):
        nx, ny = x + DX[i], y + DY[i]
        if is_valid(nx, ny, m, n) and not visited[flat(nx, ny, n)]:
            cnt += 1
    return cnt
def choose_start(m: int, n: int, rng: random.Random) -> int:
    inner = [
        flat(x, y, n)
        for x in range(1, m - 1)
        for y in range(1, n - 1)
    ]
    pool = inner if inner else [flat(x, y, n)
                                for x in range(m) for y in range(n)]
    return rng.choice(pool)
def connectivity_and_parity_ok(
    visited:    list[bool],
    seed_flat:  int,
    expected:   int,
    m: int, n: int,
    gs_flat:    int,
    seen_buf:   list[bool],     
    seen_cells: list[int],      
) -> bool:
    if expected == 0:
        return True
    sx, sy = coords(seed_flat, n)
    gx, gy = coords(gs_flat,   n)
    seen_buf[seed_flat] = True
    seen_cells.append(seed_flat)
    q = deque([seed_flat])
    whites = 1 if (sx + sy) % 2 == 0 else 0
    blacks = 1 - whites
    near_start = (abs(sx - gx) + abs(sy - gy) == 1)
    count = 0
    while q:
        cf = q.popleft()
        count += 1
        cx, cy = coords(cf, n)
        for i in range(4):
            nx, ny = cx + DX[i], cy + DY[i]
            if not is_valid(nx, ny, m, n):
                continue
            nf = flat(nx, ny, n)
            if visited[nf] or seen_buf[nf]:
                continue
            seen_buf[nf] = True
            seen_cells.append(nf)
            q.append(nf)
            if (nx + ny) % 2 == 0:
                whites += 1
            else:
                blacks += 1
            if not near_start and abs(nx - gx) + abs(ny - gy) == 1:
                near_start = True
    for cf in seen_cells:
        seen_buf[cf] = False
    seen_cells.clear()
    return (count == expected
            and near_start
            and abs(whites - blacks) <= 1)
def dfs_iterative(
    m: int, n: int,
    start_flat: int,
    rng:         random.Random,
    warnsdorff_weight: float,   
    random_weight:     float,   
) -> list[int] | None:
    total   = m * n
    visited = [False] * total
    visited[start_flat] = True
    seen_buf   = [False] * total
    seen_cells: list[int] = []
    def get_neighbors(f: int) -> list[tuple[int, float]]:
        x, y = coords(f, n)
        nbrs = []
        for i in range(4):
            nx, ny = x + DX[i], y + DY[i]
            if is_valid(nx, ny, m, n):
                nf = flat(nx, ny, n)
                if not visited[nf]:
                    wf = count_free_neighbors(nf, visited, m, n)
                    score = (warnsdorff_weight * wf
                             + random_weight * rng.random())
                    nbrs.append((nf, score))
        nbrs.sort(key=lambda t: t[1])
        return nbrs
    stack: list[list] = [[start_flat, get_neighbors(start_flat), 0]]
    path:  list[int]  = [start_flat]
    while stack:
        frame        = stack[-1]
        f, nbrs, idx = frame[0], frame[1], frame[2]
        depth        = len(path)
        if depth == total:
            x,  y  = coords(f,          n)
            gx, gy = coords(start_flat, n)
            if abs(x - gx) + abs(y - gy) == 1:
                path.append(start_flat)   
                return path
            stack.pop()
            if f != start_flat:
                path.pop()
                visited[f] = False
            continue
        if idx < len(nbrs):
            nf, _ = nbrs[idx]
            frame[2] += 1
            if visited[nf]:
                continue
            visited[nf] = True
            remaining   = total - depth - 1
            can_go      = True
            if remaining > 0:
                nx, ny    = coords(nf, n)
                seed      = None
                for i in range(4):
                    tx, ty = nx + DX[i], ny + DY[i]
                    if is_valid(tx, ty, m, n):
                        tf = flat(tx, ty, n)
                        if not visited[tf]:
                            seed = tf
                            break
                if seed is None:
                    can_go = False
                else:
                    can_go = connectivity_and_parity_ok(
                        visited, seed, remaining,
                        m, n, start_flat,
                        seen_buf, seen_cells
                    )
            if can_go:
                path.append(nf)
                stack.append([nf, get_neighbors(nf), 0])
            else:
                visited[nf] = False
        else:
            stack.pop()
            if f != start_flat:
                path.pop()
                visited[f] = False
    return None
def find_hamiltonian_cycle(
    m: int, n: int,
    max_restarts:      int   = 200,
    time_limit_sec:    float = 300.0,
    warnsdorff_weight: float = 1.0,
    random_weight:     float = 0.5,
    seed:              int | None = None,
    verbose:           bool  = True,
) -> tuple[list[int] | None, dict]:
    if m % 2 == 1 and n % 2 == 1:
        if verbose:
            print(f"[WARN] Grid {m}x{n}: odd number of cells — "
                  f"Hamiltonian cycle impossible. n reduced to {n-1}.")
        n -= 1
    rng        = random.Random(seed)
    t0         = time.perf_counter()
    stats      = {
        "restarts": 0, "total_time": 0.0,
        "m": m, "n": n, "found": False
    }
    for attempt in range(max_restarts):
        elapsed = time.perf_counter() - t0
        if elapsed >= time_limit_sec:
            if verbose:
                print(f"[STOP] Time limit exceeded ({time_limit_sec:.0f}s).")
            break
        adaptive_rw = random_weight * (1 + attempt * 0.05)
        start = choose_start(m, n, rng)
        sx, sy = coords(start, n)
        if verbose:
            print(f"  Restart {attempt+1:>4}/{max_restarts} | "
                  f"start=({sx},{sy}) | "
                  f"rw={adaptive_rw:.2f} | "
                  f"elapsed={elapsed:.1f}s", end="\r")
        result = dfs_iterative(
            m, n, start, rng,
            warnsdorff_weight, adaptive_rw
        )
        stats["restarts"] = attempt + 1
        if result is not None:
            stats["total_time"] = time.perf_counter() - t0
            stats["found"]      = True
            if verbose:
                print(f"\n[OK]  Found in {stats['total_time']:.2f}s, "
                      f"restarts: {stats['restarts']}")
            return result, stats
    stats["total_time"] = time.perf_counter() - t0
    if verbose:
        print(f"\n[FAIL] Not found. Restarts: {stats['restarts']}, "
              f"time: {stats['total_time']:.2f}s")
    return None, stats
def visualize(
    cycle:  list[int],
    m: int, n: int,
    stats:  dict,
    title:  str = "Hamiltonian Cycle",
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(6, m * 0.7)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    heat = np.full((m, n), np.nan)
    path_coords = [coords(f, n) for f in cycle]
    for step, (x, y) in enumerate(path_coords[:-1]):   
        heat[x][y] = step
    im = ax.imshow(heat, cmap="plasma", origin="upper",
                   vmin=0, vmax=m * n, alpha=0.85)
    ys = [c[1] for c in path_coords]
    xs = [c[0] for c in path_coords]
    ax.plot(ys, xs, color="#00ffcc", linewidth=1.2,
            alpha=0.9, zorder=3)
    ax.scatter(ys, xs, color="#ffffff", s=15, zorder=4)
    k = max(1, len(path_coords) // 40)
    for i in range(0, len(path_coords) - 1, k):
        x0, y0 = path_coords[i]
        x1, y1 = path_coords[i + 1]
        ax.annotate(
            "", xy=(y1, x1), xytext=(y0, x0),
            arrowprops=dict(
                arrowstyle="->",
                color="#ffffff",
                lw=0.8,
            ),
            zorder=4,
        )
    
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="#1e2a38",
            linestyle="-", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(m))
    ax.tick_params(colors="#4a6080", labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Step", color="#8899aa", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#8899aa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8899aa")
    info = (f"{title}  |  {m}x{n}  |  "
            f"restarts: {stats['restarts']}  |  "
            f"time: {stats['total_time']:.2f}s")
    ax.set_title(info, color="#cdd9e5", fontsize=11, pad=12)
    legend = ax.legend(facecolor="#1e2a38", edgecolor="#4a6080",
                       labelcolor="#cdd9e5", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"hamiltonian_{m}x{n}.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[INFO] Plot saved: hamiltonian_{m}x{n}.png")
    plt.show()
def visualize_animation(
    cycle:  list[int],
    m: int, n: int,
    interval_ms: int = 30,
) -> None:
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(6, m * 0.7)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    heat = np.zeros((m, n))
    im   = ax.imshow(heat, cmap="plasma", origin="upper",
                     vmin=0, vmax=m * n, alpha=0.85)
    line, = ax.plot([], [], color="#00ffcc", linewidth=1.5,
                    alpha=0.9, zorder=3)
    path_coords = [coords(f, n) for f in cycle]
    
    # Draw all vertices as dots
    all_ys = [c[1] for c in path_coords]
    all_xs = [c[0] for c in path_coords]
    ax.scatter(all_ys, all_xs, color="#ffffff", s=10, zorder=2, alpha=0.6)

    total       = len(path_coords)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
    ax.grid(which="minor", color="#1e2a38",
            linestyle="-", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    title_text = ax.set_title(f"Animation: {m}x{n}",
                 color="#cdd9e5", fontsize=11, pad=12)
    state = {'interval': interval_ms, 'running': True, 'frame': 0}
    def update(frame):
        state['frame'] = frame
        x, y = path_coords[frame]
        heat[x][y] = frame + 1
        im.set_data(heat)
        ys = [c[1] for c in path_coords[: frame + 1]]
        xs = [c[0] for c in path_coords[: frame + 1]]
        line.set_data(ys, xs)
        pct = (frame + 1) / total * 100
        status = "PLAY" if state['running'] else "PAUSE"
        title_text.set_text(f"Animation: {m}x{n} | {pct:.0f}% | {status} | Delay: {state['interval']:.0f}ms\n[Space]=Pause/Resume, [Up/Down]=Speed")
        return im, line, title_text
    anim = FuncAnimation(
        fig, update, frames=total,
        interval=interval_ms, blit=True, repeat=False
    )
    def on_key(event):
        if event.key == ' ':
            if state['running']:
                anim.event_source.stop()
                state['running'] = False
            else:
                anim.event_source.start()
                state['running'] = True
        elif event.key == 'up':
            state['interval'] = max(1, state['interval'] / 2.0)
            anim.event_source.interval = int(state['interval'])
            if state['running']:
                anim.event_source.stop()
                anim.event_source.start()
        elif event.key == 'down':
            state['interval'] = min(3000, state['interval'] * 2.0)
            anim.event_source.interval = int(state['interval'])
            if state['running']:
                anim.event_source.stop()
                anim.event_source.start()
        status = "PLAY" if state['running'] else "PAUSE"
        pct = (state['frame'] + 1) / total * 100
        title_text.set_text(f"Animation: {m}x{n} | {pct:.0f}% | {status} | Delay: {state['interval']:.0f}ms\n[Space]=Pause/Resume, [Up/Down]=Speed")
        if not state['running']:
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
def main():
    print("=" * 60)
    print("  Finding Hamiltonian Cycle in Grid Graph")
    print("=" * 60)
    try:
        line = input("\nEnter grid size (m n): ").strip()
        if not line:
            return
        m, n = map(int, line.split())
        print("\nSearch Parameters (Enter = default):")
        max_r = input(f"  Max Restarts         [200]:  ").strip()
        max_restarts = int(max_r) if max_r else 200
        t_lim = input(f"  Time Limit (sec)     [300]:  ").strip()
        time_limit = float(t_lim) if t_lim else 300.0
        ww = input(f"  Warnsdorff Weight    [1.0]:  ").strip()
        warnsdorff_weight = float(ww) if ww else 1.0
        rw = input(f"  Randomness Weight    [0.5]:  ").strip()
        random_weight = float(rw) if rw else 0.5
        sd = input(f"  Seed (for reprod.)   [None]: ").strip()
        seed = int(sd) if sd else None
        anim = input(f"  Animation?           [n/y]:  ").strip().lower()
    except (ValueError, EOFError):
        print("[ERR] Invalid input.")
        return
    print()
    cycle, stats = find_hamiltonian_cycle(
        m, n,
        max_restarts      = max_restarts,
        time_limit_sec    = time_limit,
        warnsdorff_weight = warnsdorff_weight,
        random_weight     = random_weight,
        seed              = seed,
        verbose           = True,
    )
    if cycle is None:
        print("\nHamiltonian cycle not found within time/restarts limit.")
        return
    actual_n = stats["n"]
    assert len(cycle) == m * actual_n + 1,  "Invalid path length"
    assert cycle[0] == cycle[-1],           "Cycle not closed"
    visited_check = set(cycle[:-1])
    assert len(visited_check) == m * actual_n, "Not all cells visited"
    print("[OK] Cycle verified.")
    if anim == "y":
        visualize_animation(cycle, m, actual_n)
    else:
        visualize(cycle, m, actual_n, stats)
if __name__ == "__main__":
    main()