"""P0: Strict vs Tolerant parser analysis for ALL 8 tasks x 4 strategies."""

import json, ast, os, statistics
import networkx as nx


def parse_strict(raw_text):
    text = raw_text.strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(parsed, tuple) or len(parsed) != 2:
        return None
    n, raw_edges = parsed
    if not isinstance(n, int) or n < 0:
        return None
    if not isinstance(raw_edges, (list, tuple)):
        return None
    edges = []
    seen = set()
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return None
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int):
            return None
        if u < 0 or v < 0 or u >= n or v >= n:
            return None
        canon = (u, v) if u <= v else (v, u)
        if canon not in seen:
            seen.add(canon)
            edges.append(canon)
    return (n, edges)


def parse_tolerant(raw_text):
    text = raw_text.strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(parsed, tuple) or len(parsed) != 2:
        return None
    n, raw_edges = parsed
    if isinstance(n, str):
        try:
            n = int(n)
        except ValueError:
            return None
    if not isinstance(n, int) or n < 0:
        return None
    if not isinstance(raw_edges, (list, tuple)):
        return None
    edges = []
    seen = set()
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return None
        u, v = edge
        if isinstance(u, str):
            try:
                u = int(u)
            except ValueError:
                return None
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                return None
        if not isinstance(u, int) or not isinstance(v, int):
            return None
        if u < 0 or v < 0 or u >= n or v >= n:
            return None
        canon = (u, v) if u <= v else (v, u)
        if canon not in seen:
            seen.add(canon)
            edges.append(canon)
    return (n, edges)


# ========== Validators ==========
def is_connected(n, edges):
    if n == 0:
        return True
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return len(visited) == n


def comp_count(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = [False] * n
    count = 0
    for s in range(n):
        if visited[s]:
            continue
        count += 1
        stack = [s]
        visited[s] = True
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
    return count


def validate(task_id, n, edges, task_params):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    if task_id == "tree":
        return is_connected(n, edges) and len(edges) == n - 1
    elif task_id == "cycle":
        if n < 3:
            return False
        if not is_connected(n, edges):
            return False
        if len(edges) != n:
            return False
        return all(len(adj[i]) == 2 for i in range(n))
    elif task_id == "planar":
        ok, _ = nx.check_planarity(G)
        m_expected = task_params.get("m", 0)
        if m_expected > 0 and len(edges) != m_expected:
            ok = False
        return ok
    elif task_id == "components":
        return comp_count(n, edges) == task_params["k"]
    elif task_id == "k_regular":
        return all(len(adj[i]) == task_params["k"] for i in range(n))
    elif task_id == "wheel":
        if n < 4:
            return False
        centers = [i for i in range(n) if len(adj[i]) == n - 1]
        if len(centers) != 1:
            return False
        center = centers[0]
        rim = [i for i in range(n) if i != center]
        for nd in rim:
            if len(adj[nd]) != 3:
                return False
        rim_edges = [(u, v) for u, v in edges if u != center and v != center]
        mapping = {old: new for new, old in enumerate(rim)}
        re_edges = []
        for u, v in rim_edges:
            a, b = mapping[u], mapping[v]
            re_edges.append((a, b) if a <= b else (b, a))
        rn = n - 1
        if rn < 3:
            return False
        radj = [set() for _ in range(rn)]
        for u, v in re_edges:
            radj[u].add(v)
            radj[v].add(u)
        if len(re_edges) != rn:
            return False
        if not all(len(radj[i]) == 2 for i in range(rn)):
            return False
        vis = {0}
        st = [0]
        while st:
            c = st.pop()
            for nx_ in radj[c]:
                if nx_ not in vis:
                    vis.add(nx_)
                    st.append(nx_)
        return len(vis) == rn
    elif task_id == "bipartite":
        color = [-1] * n
        for i in range(n):
            if color[i] != -1:
                continue
            queue = [i]
            color[i] = 0
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True
    elif task_id == "k_coloring":
        m_expected = task_params.get("m", 0)
        if m_expected > 0 and len(edges) != m_expected:
            return False
        k = task_params["k"]
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        num_colors = max(coloring.values()) + 1 if coloring else 0
        return num_colors <= k
    return False


# ========== Config ==========
TASK_PARAMS = {
    "tree": {"n": 8},
    "cycle": {"n": 10},
    "planar": {"n": 8, "m": 12},
    "components": {"n": 8, "k": 3},
    "k_regular": {"n": 12, "k": 3},
    "wheel": {"n": 8},
    "bipartite": {"n": 6, "k": 3},
    "k_coloring": {"n": 10, "m": 20, "k": 3},
}
TASK_ORDER = ["tree", "cycle", "planar", "components", "k_regular", "wheel", "bipartite", "k_coloring"]
TASK_NAMES = {
    "tree": "Tree", "cycle": "Cycle", "planar": "Planar",
    "components": "#Components", "k_regular": "k-regular", "wheel": "Wheel",
    "bipartite": "Bipartite", "k_coloring": "k-coloring",
}

base = os.path.expanduser("~/Desktop/llm4graphgen-results")
strategies = {
    "zero_shot": "stage2_Llama-2-13b-chat-hf_zero_shot_small_100s_r3_20260302_190154",
    "few_shot": "stage2_Llama-2-13b-chat-hf_few_shot_small_100s_r3_20260302_203933",
    "zero_shot_cot": "stage2_Llama-2-13b-chat-hf_zero_shot_cot_small_100s_r3_20260302_221130",
    "few_shot_cot": "stage2_Llama-2-13b-chat-hf_few_shot_cot_small_100s_r3_20260303_062131",
}


def check_has_quotes(raw):
    """Check if a parsed output contains string-quoted numbers."""
    try:
        parsed_raw = ast.literal_eval(raw.strip())
    except Exception:
        return False
    if not isinstance(parsed_raw, tuple) or len(parsed_raw) != 2:
        return False
    n_val, edges_val = parsed_raw
    if isinstance(n_val, str):
        return True
    if isinstance(edges_val, (list, tuple)):
        for e in edges_val:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                if isinstance(e[0], str) or isinstance(e[1], str):
                    return True
    return False


def main():
    header = f"{'Task':<14} {'Strategy':<16} | {'Tolerant%':>10} {'Strict%':>10} {'Diff(pp)':>10} | {'Quoted':>6} {'Total':>6} {'Quote%':>7}"
    print("=" * 100)
    print(header)
    print("=" * 100)

    all_results = {}

    for task_id in TASK_ORDER:
        tp = TASK_PARAMS[task_id]
        for strat_name, dirname in strategies.items():
            tolerant_rates = []
            strict_rates = []
            quoted_total = 0
            total_total = 0

            for r in [1, 2, 3]:
                fpath = os.path.join(base, dirname, f"llm_io_r{r}.jsonl")
                with open(fpath, encoding="utf-8") as f:
                    lines = f.readlines()

                tolerant_valid = 0
                strict_valid = 0
                quoted_count = 0
                task_count = 0

                for line in lines:
                    rec = json.loads(line)
                    if rec["task_id"] != task_id:
                        continue
                    task_count += 1
                    raw = rec["raw_output"]

                    if check_has_quotes(raw):
                        quoted_count += 1

                    t_result = parse_tolerant(raw)
                    if t_result:
                        n_t, edges_t = t_result
                        if validate(task_id, n_t, edges_t, tp):
                            tolerant_valid += 1

                    s_result = parse_strict(raw)
                    if s_result:
                        n_s, edges_s = s_result
                        if validate(task_id, n_s, edges_s, tp):
                            strict_valid += 1

                if task_count > 0:
                    tolerant_rates.append(tolerant_valid / task_count * 100)
                    strict_rates.append(strict_valid / task_count * 100)
                quoted_total += quoted_count
                total_total += task_count

            t_mean = statistics.mean(tolerant_rates) if tolerant_rates else 0
            s_mean = statistics.mean(strict_rates) if strict_rates else 0
            t_std = statistics.stdev(tolerant_rates) if len(tolerant_rates) >= 2 else 0
            s_std = statistics.stdev(strict_rates) if len(strict_rates) >= 2 else 0
            diff = t_mean - s_mean
            q_pct = quoted_total / total_total * 100 if total_total > 0 else 0

            all_results[(task_id, strat_name)] = {
                "tolerant": t_mean, "strict": s_mean, "diff": diff,
                "quoted_pct": q_pct, "t_std": t_std, "s_std": s_std,
            }

            marker = " ***" if abs(diff) >= 5 else (" *" if abs(diff) >= 1 else "")
            print(f"{TASK_NAMES[task_id]:<14} {strat_name:<16} | {t_mean:>9.1f}% {s_mean:>9.1f}% {diff:>+9.1f}pp | {quoted_total:>5}  {total_total:>5}  {q_pct:>5.1f}%{marker}")
        print("-" * 100)

    # Summary
    print()
    print("SUMMARY: Tasks with significant tolerance impact (diff >= 1pp in any strategy):")
    print("-" * 80)
    affected = {}
    for task_id in TASK_ORDER:
        max_diff = 0
        details = []
        for strat_name in strategies:
            r = all_results[(task_id, strat_name)]
            if r["diff"] >= 1:
                details.append(f"{strat_name}: +{r['diff']:.1f}pp")
                max_diff = max(max_diff, r["diff"])
        if details:
            print(f"  {TASK_NAMES[task_id]:<14}: {' | '.join(details)}")
            affected[task_id] = max_diff
    if not affected:
        print("  (none)")

    print()
    print("Tasks with NO tolerance impact (all strategies diff < 1pp):")
    for task_id in TASK_ORDER:
        if task_id not in affected:
            print(f"  {TASK_NAMES[task_id]}")

    # Detailed table for report update
    print()
    print("=" * 100)
    print("DETAILED TABLE (for report): Strict Valid Rate by Task x Strategy")
    print("=" * 100)
    print(f"{'Task':<14} | {'ZS-tol':>7} {'ZS-str':>7} | {'FS-tol':>7} {'FS-str':>7} | {'ZSC-tol':>7} {'ZSC-str':>7} | {'FSC-tol':>7} {'FSC-str':>7}")
    print("-" * 100)
    strat_keys = ["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"]
    for task_id in TASK_ORDER:
        parts = []
        for sk in strat_keys:
            r = all_results[(task_id, sk)]
            parts.append(f"{r['tolerant']:>6.1f}% {r['strict']:>6.1f}%")
        print(f"{TASK_NAMES[task_id]:<14} | {' | '.join(parts)}")


if __name__ == "__main__":
    main()
