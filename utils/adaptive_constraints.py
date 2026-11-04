"""Global score ordering with sequential feasibility and sparse thin topology.

Scores are never normalized, augmented, or ranked inside instances. Quotas are
updated after every accepted removal. Sparse graph connectivity is checked on
the graph *after* earlier accepted removals, not from a stale articulation mask.
A bounded connectivity search may reject a safe removal conservatively; it never
approves a removal merely because the search budget was exhausted.
"""
import time

import numpy as np


def _array(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _counts(ids):
    values, count = np.unique(ids, return_counts=True)
    return {int(k): int(v) for k, v in zip(values, count)}


def _quota_dict(value):
    result = {} if value is None else {int(k): int(v) for k, v in value.items()}
    if any(v < 0 for v in result.values()):
        raise ValueError("Survival quotas must be nonnegative")
    return result


class NeighborIndex:
    """Batched CPU KNN; exact cKDTree when installed, FLANN otherwise.

    The optional FLANN backend is approximate and this is reported in topology
    provenance. The topology guarantee concerns the constructed sparse graph,
    not the unknown continuous scene topology or exact Euclidean KNN graph.
    """
    def __init__(self, xyz, backend="auto", trees=4, checks=48):
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError("Neighbor positions must be finite [N,3]")
        if not len(self.xyz):
            raise ValueError("Cannot index an empty point set")
        self.checks = int(checks)
        self.backend = backend
        if backend in ("auto", "scipy"):
            try:
                from scipy.spatial import cKDTree
                self.index = cKDTree(self.xyz)
                self.backend = "scipy_ckdtree_exact"
            except ImportError:
                if backend == "scipy":
                    raise
        if self.backend in ("auto", "flann"):
            import cv2
            cv2.setRNGSeed(0)
            self.index = cv2.flann_Index(self.xyz, dict(algorithm=1, trees=int(trees)))
            self.backend = "opencv_flann_approximate"
        if self.backend not in ("scipy_ckdtree_exact", "opencv_flann_approximate"):
            raise ValueError("Unknown KNN backend " + str(backend))

    def query(self, xyz, k):
        query = np.ascontiguousarray(xyz, dtype=np.float32)
        k = min(max(int(k), 1), len(self.xyz))
        if self.backend == "scipy_ckdtree_exact":
            dist, ids = self.index.query(query, k=k)
            return np.asarray(ids).reshape(len(query), k).astype(np.int64), np.asarray(dist).reshape(len(query), k).astype(np.float32)
        ids, squared = self.index.knnSearch(query, k, params=dict(checks=self.checks))
        return ids.astype(np.int64), np.sqrt(np.maximum(squared, 0))


def graph_from_edges(node_ids, edges):
    """Build undirected CSR. Edges index rows of node_ids, not global IDs."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    count = len(node_ids)
    if len(np.unique(node_ids)) != count or np.any(node_ids < 0):
        raise ValueError("Graph node IDs must be unique nonnegative indices")
    edge = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edge) and (np.any(edge < 0) or np.any(edge >= count)):
        raise ValueError("Invalid local edge endpoint")
    edge = edge[edge[:, 0] != edge[:, 1]]
    if len(edge):
        code = np.concatenate((edge[:, 0] * count + edge[:, 1], edge[:, 1] * count + edge[:, 0]))
        code = np.unique(code)
        rows, neighbors = code // count, code % count
        offsets = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=count))))
    else:
        offsets = np.zeros(count + 1, dtype=np.int64)
        neighbors = np.zeros(0, dtype=np.int64)
    return dict(node_ids=node_ids, offsets=offsets.astype(np.int64), neighbors=neighbors.astype(np.int32))


def _validate_graph(graph, total_count=None):
    ids = np.asarray(graph["node_ids"], dtype=np.int64)
    offsets = np.asarray(graph["offsets"], dtype=np.int64)
    neighbors = np.asarray(graph["neighbors"], dtype=np.int64)
    if offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(neighbors) or np.any(np.diff(offsets) < 0):
        raise ValueError("Malformed topology CSR")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or (total_count is not None and np.any(ids >= total_count)):
        raise ValueError("Topology global IDs are invalid")
    if np.any(neighbors < 0) or np.any(neighbors >= len(ids)):
        raise ValueError("Topology local neighbor IDs are invalid")
    return ids, offsets, neighbors


def articulation_points(graph, active=None):
    """Iterative Tarjan in O(V+E), with no recursion-depth dependency."""
    ids, offsets, neighbors = _validate_graph(graph)
    size = len(ids)
    active = np.ones(size, bool) if active is None else np.asarray(active, bool)
    if active.shape != (size,):
        raise ValueError("Graph active mask has incorrect length")
    discovery = np.full(size, -1, np.int64)
    low = np.empty(size, np.int64)
    parent = np.full(size, -1, np.int64)
    child_count = np.zeros(size, np.int32)
    critical = np.zeros(size, bool)
    tick = 0
    component_count = 0
    for root in np.flatnonzero(active):
        if discovery[root] >= 0:
            continue
        component_count += 1
        discovery[root] = low[root] = tick
        tick += 1
        stack = [(int(root), int(offsets[root]))]
        while stack:
            node, next_edge = stack[-1]
            if next_edge < offsets[node + 1]:
                other = int(neighbors[next_edge])
                stack[-1] = (node, next_edge + 1)
                if not active[other]:
                    continue
                if discovery[other] < 0:
                    parent[other] = node
                    child_count[node] += 1
                    discovery[other] = low[other] = tick
                    tick += 1
                    stack.append((other, int(offsets[other])))
                elif other != parent[node]:
                    low[node] = min(low[node], discovery[other])
            else:
                stack.pop()
                previous = parent[node]
                if previous < 0:
                    critical[node] = child_count[node] > 1
                else:
                    low[previous] = min(low[previous], low[node])
                    if parent[previous] >= 0 and low[node] >= discovery[previous]:
                        critical[previous] = True
    return critical, dict(nodes=int(active.sum()), edges=int(sum(np.count_nonzero(active[neighbors[offsets[i]:offsets[i + 1]]]) for i in np.flatnonzero(active)) // 2), components=int(component_count), articulation_points=int(critical.sum()))


def build_topology(xyz, instance_ids, confidence=None, boundary_mask=None, config=None):
    """Batched local PCA over all points, then a candidate-only sparse graph.

    Reliable same-instance neighbors are preferred for PCA whenever >=3 are
    found in the oversampled neighborhood. Unknown neighbors fall back to local
    geometry. Graph edges never join two different reliable instances. Strong
    boundary endpoints are excluded by default; their protection uses quotas.
    Default edge radius is distance_multiplier times the larger endpoint local
    nearest spacing. A finite distance_threshold instead sets an absolute cap.
    """
    started = time.time()
    cfg = {} if config is None else dict(config)
    xyz = _array(xyz, np.float32)
    ids = _array(instance_ids, np.int64)
    size = len(xyz)
    if xyz.shape != (size, 3) or ids.shape != (size,) or not np.isfinite(xyz).all():
        raise ValueError("Topology inputs are not aligned finite positions/IDs")
    conf = np.ones(size, np.float32) if confidence is None else _array(confidence, np.float32)
    boundary = np.zeros(size, bool) if boundary_mask is None else _array(boundary_mask, bool)
    if conf.shape != (size,) or boundary.shape != (size,):
        raise ValueError("Topology confidence/boundary arrays are not aligned")
    reliable = (ids >= 0) & (conf >= float(cfg.get("confidence_threshold", .5)))
    linearity = np.zeros(size, np.float32)
    direction = np.zeros((size, 3), np.float32)
    spacing = np.zeros(size, np.float32)
    if size < 3 or not cfg.get("enabled", True):
        empty = graph_from_edges([], [])
        return dict(linearity=linearity, direction=direction, principal_direction=direction,
                    thin_candidate_mask=np.zeros(size, bool), critical_mask=np.zeros(size, bool),
                    topology_critical_mask=np.zeros(size, bool), graph=empty,
                    audit=dict(nodes=0, edges=0, components=0, articulation_points=0, seconds=time.time()-started, knn_backend="none"))
    knn = max(3, int(cfg.get("knn", 12)))
    batch_size = int(cfg.get("batch_size", 8192))
    if batch_size <= 0:
        raise ValueError("Topology batch_size must be positive")
    index = NeighborIndex(xyz, backend=cfg.get("knn_backend", "auto"), trees=cfg.get("flann_trees", 4), checks=cfg.get("flann_checks", 48))
    query_k = min(size, int(cfg.get("knn_oversampling", 2)) * knn + 1)
    preferred_count = 0
    for begin in range(0, size, batch_size):
        end = min(size, begin + batch_size)
        neighbors, distances = index.query(xyz[begin:end], query_k)
        row_ids = np.arange(begin, end)
        not_self = neighbors != row_ids[:, None]
        same = not_self & reliable[row_ids, None] & reliable[neighbors] & (ids[neighbors] == ids[row_ids, None])
        use_same = same.sum(axis=1) >= 3
        preferred_count += int(use_same.sum())
        eligible = np.where(use_same[:, None], same, not_self)
        # Sorted distances remain the primary order inside each eligibility set.
        rank = np.where(eligible, distances, np.inf)
        take = np.argsort(rank, axis=1, kind="stable")[:, :min(knn, query_k - 1)]
        selected = np.take_along_axis(neighbors, take, axis=1)
        valid = np.take_along_axis(eligible, take, axis=1)
        weight = valid.astype(np.float32)
        local = xyz[selected].astype(np.float64)
        total = np.maximum(weight.sum(axis=1), 1)
        center = (local * weight[..., None]).sum(axis=1) / total[:, None]
        centered = (local - center[:, None, :]) * weight[..., None]
        covariance = np.einsum("nki,nkj->nij", centered, centered) / total[:, None, None]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        linearity[begin:end] = np.maximum(0, (eigenvalues[:, 2] - eigenvalues[:, 1]) / np.maximum(eigenvalues[:, 2], 1e-20))
        direction[begin:end] = eigenvectors[:, :, 2]
        positive = np.where(eligible & (distances > 1e-10), distances, np.inf)
        nearest = positive.min(axis=1)
        nearest[~np.isfinite(nearest)] = 0
        spacing[begin:end] = nearest
    backend = index.backend
    del index
    candidate = linearity > float(cfg.get("linearity_threshold", .80))
    node_ids = np.flatnonzero(candidate)
    all_edges = []
    if len(node_ids) > 1:
        index = NeighborIndex(xyz[node_ids], backend=cfg.get("knn_backend", "auto"), trees=cfg.get("flann_trees", 4), checks=cfg.get("flann_checks", 48))
        graph_k = min(len(node_ids), int(cfg.get("graph_knn", knn)) + 1)
        for begin in range(0, len(node_ids), batch_size):
            end = min(len(node_ids), begin + batch_size)
            neighbor_local, distances = index.query(xyz[node_ids[begin:end]], graph_k)
            source_local = np.broadcast_to(np.arange(begin, end)[:, None], neighbor_local.shape)
            source = node_ids[source_local]
            target = node_ids[neighbor_local]
            admissible = source_local != neighbor_local
            threshold = cfg.get("distance_threshold")
            if threshold is None:
                radius = float(cfg.get("distance_multiplier", 2.5)) * np.maximum(spacing[source], spacing[target])
            else:
                radius = float(threshold)
            admissible &= (distances < radius) & (distances > 1e-10)
            admissible &= np.abs((direction[source] * direction[target]).sum(axis=2)) > float(cfg.get("direction_threshold", .7))
            admissible &= ~(reliable[source] & reliable[target] & (ids[source] != ids[target]))
            if cfg.get("exclude_boundary_edges", True):
                admissible &= ~(boundary[source] | boundary[target])
            all_edges.append(np.column_stack((source_local[admissible], neighbor_local[admissible])))
        del index
    graph = graph_from_edges(node_ids, np.concatenate(all_edges) if all_edges else [])
    local_critical, graph_audit = articulation_points(graph)
    critical = np.zeros(size, bool)
    critical[node_ids] = local_critical
    graph_audit.update(seconds=float(time.time()-started), knn_backend=backend,
                       pca_points=size, same_instance_pca_points=preferred_count,
                       candidate_points=int(candidate.sum()), boundary_edge_exclusion=bool(cfg.get("exclude_boundary_edges", True)),
                       scope="Connectivity of constructed sparse candidate graph; not a guarantee of rendered or physical topology")
    return dict(linearity=linearity, direction=direction, principal_direction=direction, local_spacing=spacing,
                thin_candidate_mask=candidate, critical_mask=critical, topology_critical_mask=critical,
                graph=graph, audit=graph_audit)


class DynamicConnectivityGuard:
    """Sequential vertex deletion with a conservative bounded connectivity test.

    A vertex can be deleted iff its live neighbors are connected without it.
    This equivalent local criterion prevents any increase in connected-component
    count caused by a deletion. Full-component searches can be expensive, so the
    search stops after max_search_nodes and rejects that removal explicitly.
    Isolated vertices may disappear; no component-size or endpoint guarantee is
    implied. Recomputing topology each compression round refreshes the geometry.
    """
    def __init__(self, graph, total_count, active_mask=None, max_search_nodes=4096):
        ids, offsets, neighbors = _validate_graph(graph, total_count)
        self.node_ids, self.offsets, self.neighbors = ids, offsets, neighbors
        self.lookup = np.full(total_count, -1, np.int64)
        self.lookup[ids] = np.arange(len(ids))
        self.active = np.ones(len(ids), bool) if active_mask is None else np.asarray(active_mask, bool)[ids].copy()
        self.max_search_nodes = int(max_search_nodes)
        if self.max_search_nodes < 1:
            raise ValueError("Connectivity search limit must be >=1")
        self.search_limited = 0
        self.connectivity_rejections = 0
        self.checks = 0

    def live_neighbors(self, node):
        neighbors = self.neighbors[self.offsets[node]:self.offsets[node+1]]
        return neighbors[self.active[neighbors]]

    def can_remove(self, global_id):
        node = self.lookup[int(global_id)]
        if node < 0:
            return True
        if not self.active[node]:
            return False
        self.checks += 1
        adjacent = self.live_neighbors(node)
        if len(adjacent) <= 1:
            return True
        targets = set(map(int, adjacent[1:]))
        visited = {int(adjacent[0])}
        queue = [int(adjacent[0])]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for other in self.neighbors[self.offsets[current]:self.offsets[current+1]]:
                other = int(other)
                if other == node or not self.active[other] or other in visited:
                    continue
                targets.discard(other)
                if not targets:
                    return True
                visited.add(other)
                queue.append(other)
                if len(visited) >= self.max_search_nodes:
                    self.search_limited += 1
                    return False
        self.connectivity_rejections += 1
        return False

    def remove(self, global_id):
        node = self.lookup[int(global_id)]
        if node >= 0:
            self.active[node] = False


def select_feasible_prune(scores, instance_ids, instance_min_counts, remove_count,
                          boundary_mask=None, boundary_min_counts=None,
                          boundary_mode="quota", topology=None, active_mask=None,
                          max_search_nodes=4096, raise_on_infeasible=False):
    """Select removals in unchanged ascending GLOBAL score order.

    The round-start articulation mask is hard-protected for the whole call.
    Dynamic quotas and additional graph checks are evaluated immediately before
    deletion; a former articulation is never released after its leaves vanish.
    One pass skips currently infeasible candidates and continues globally.
    Infeasibility is explicit in the returned audit and never relaxes a quota or
    graph constraint. Unknown (-1) instance IDs do not receive instance quotas,
    but can receive their own boundary quota via boundary_min_counts[-1].
    """
    started = time.time()
    score = _array(scores, np.float64).reshape(-1)
    ids = _array(instance_ids, np.int64)
    size = len(score)
    if ids.shape != (size,) or not np.isfinite(score).all():
        raise ValueError("Scores must be finite and aligned with instance IDs")
    requested = int(remove_count)
    if requested < 0:
        raise ValueError("Requested removal count must be nonnegative")
    active = np.ones(size, bool) if active_mask is None else _array(active_mask, bool)
    boundary = np.zeros(size, bool) if boundary_mask is None else _array(boundary_mask, bool)
    if active.shape != (size,) or boundary.shape != (size,):
        raise ValueError("Active/boundary mask is not aligned")
    if boundary_mode not in ("none", "quota", "hard"):
        raise ValueError("Unknown boundary mode")
    minima = {k: v for k, v in _quota_dict(instance_min_counts).items() if k >= 0}
    boundary_minima = _quota_dict(boundary_min_counts) if boundary_mode == "quota" else {}
    current = _counts(ids[active])
    boundary_current = _counts(ids[active & boundary])
    initial_current, initial_boundary = dict(current), dict(boundary_current)
    violated = [k for k, quota in minima.items() if current.get(k, 0) < quota]
    boundary_violated = [k for k, quota in boundary_minima.items() if boundary_current.get(k, 0) < quota]
    if violated or boundary_violated:
        raise ValueError("Input state already violates quotas: instances=%s boundaries=%s" % (violated, boundary_violated))
    graph = topology.get("graph", topology) if topology is not None else None
    guard = DynamicConnectivityGuard(graph, size, active, max_search_nodes) if graph is not None else None
    initial_critical = np.zeros(size, bool)
    if topology is not None:
        supplied_critical = topology.get("critical_mask", topology.get("topology_critical_mask"))
        if supplied_critical is not None:
            initial_critical = _array(supplied_critical, bool)
            if initial_critical.shape != (size,):
                raise ValueError("Round-start topology critical mask is not aligned")
            initial_critical = initial_critical.copy()
        elif guard is not None:
            local_critical, _ = articulation_points(graph, guard.active)
            initial_critical[guard.node_ids] = local_critical
    prune = np.zeros(size, bool)
    # Stable ties resolve by original active row ID; scores themselves are read-only.
    candidates = np.flatnonzero(active)
    order = candidates[np.argsort(score[candidates], kind="stable")]
    accepted = 0
    blocked_instance = blocked_boundary = blocked_topology = scanned = 0
    blocked_initial_topology = blocked_dynamic_topology = 0
    for index in order:
        if accepted >= requested:
            break
        scanned += 1
        instance = int(ids[index])
        if instance >= 0 and current[instance] - 1 < minima.get(instance, 0):
            blocked_instance += 1
            continue
        if boundary[index] and (boundary_mode == "hard" or (boundary_mode == "quota" and boundary_current[instance] - 1 < boundary_minima.get(instance, 0))):
            blocked_boundary += 1
            continue
        if initial_critical[index]:
            blocked_topology += 1
            blocked_initial_topology += 1
            continue
        if guard is not None and not guard.can_remove(index):
            blocked_topology += 1
            blocked_dynamic_topology += 1
            continue
        prune[index] = True
        accepted += 1
        current[instance] -= 1
        if boundary[index]:
            boundary_current[instance] -= 1
        if guard is not None:
            guard.remove(index)
    unmet = requested - accepted
    audit = dict(requested_prune=requested, actual_prune=accepted,
                 count_before=int(active.sum()), count_after=int(active.sum())-accepted,
                 target_reached=unmet == 0, feasible=unmet == 0, unfilled_prune_count=unmet,
                 score_order="unchanged_global_ascending_stable", ranking_scope="global",
                 candidates_scanned=scanned, instance_protected=blocked_instance,
                 boundary_protected=blocked_boundary, topology_protected=blocked_topology,
                 topology_initial_critical_count=int(np.count_nonzero(initial_critical & active)),
                 topology_initial_protected=blocked_initial_topology,
                 topology_dynamic_protected=blocked_dynamic_topology,
                 boundary_mode=boundary_mode, seconds=float(time.time()-started),
                 instance_audit=[dict(instance_id=k, count_before=initial_current.get(k, 0), count_after=current.get(k, 0), minimum=q) for k, q in sorted(minima.items())],
                 boundary_audit=[dict(instance_id=k, count_before=initial_boundary.get(k, 0), count_after=boundary_current.get(k, 0), minimum=q) for k, q in sorted(boundary_minima.items())],
                 connectivity_search_limited=guard.search_limited if guard else 0,
                 connectivity_checks=guard.checks if guard else 0,
                 connectivity_rejections=guard.connectivity_rejections if guard else 0,
                 topology_scope="Round-start articulation mask is immutable during selection; additional sequential connectivity checks reject conservatively when search-limited")
    if unmet:
        audit["infeasibility_reason"] = "Global feasible candidates exhausted; quotas and topology constraints were not weakened"
        if raise_on_infeasible:
            raise ValueError("Pruning request infeasible: requested %d, feasible %d" % (requested, accepted))
    return prune, audit
