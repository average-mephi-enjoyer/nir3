from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Set, Tuple
import networkx as nx
from core.models import CommunicationMessage, EmployeeContext


class DynamicCommunicationGraph:
    """
    Time-decayed directed communication graph with Laplace-smoothed reciprocity.
    """

    def __init__(self, decay_window_days: int = 30):
        self.decay_window = timedelta(days=decay_window_days)
        self.edge_stream: Deque[Tuple[datetime, str, str]] = deque()
        self.adjacency_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.user_aliases: Dict[str, Set[str]] = defaultdict(set)

    def register_alias(self, user_id: str, email: str) -> None:
        u = user_id.lower().strip()
        e = email.lower().strip()
        self.user_aliases[u].add(e)
        self.user_aliases[e].add(u)

    def add_communication(self, message: CommunicationMessage) -> None:
        sender = (message.sender_user_id or message.sender_id).lower().strip()
        ts = message.timestamp
        for recipient in message.receiver_ids:
            target = recipient.lower().strip()
            if sender == target:
                continue
            self.edge_stream.append((ts, sender, target))
            self.adjacency_counts[sender][target] += 1
        self._evict_expired(ts)

    def _evict_expired(self, current_time: datetime) -> None:
        cutoff = current_time - self.decay_window
        while self.edge_stream and self.edge_stream[0][0] < cutoff:
            _, src, tgt = self.edge_stream.popleft()
            self.adjacency_counts[src][tgt] -= 1
            if self.adjacency_counts[src][tgt] <= 0:
                del self.adjacency_counts[src][tgt]
            if not self.adjacency_counts[src]:
                del self.adjacency_counts[src]

    def _resolve_candidate_nodes(self, employee_id: str) -> List[str]:
        norm = employee_id.lower().strip()
        candidates = {norm}
        for alias in self.user_aliases.get(norm, set()):
            candidates.add(alias)
        for node in list(self.adjacency_counts.keys()):
            if norm in node:
                candidates.add(node)
        return [c for c in candidates if c in self.adjacency_counts or any(c in tgts for tgts in self.adjacency_counts.values())]

    def build_ego_graph(self, employee_id: str, radius: int = 2) -> nx.DiGraph:
        g = nx.DiGraph()
        matched_nodes = self._resolve_candidate_nodes(employee_id)
        if not matched_nodes:
            g.add_node(employee_id.lower().strip())
            return g
        root_node = matched_nodes[0]
        visited_nodes: Set[str] = {root_node}
        current_layer: Set[str] = {root_node}
        for _ in range(radius):
            next_layer: Set[str] = set()
            for u in current_layer:
                for v in self.adjacency_counts.get(u, {}):
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        next_layer.add(v)
                for potential_u, targets in self.adjacency_counts.items():
                    if u in targets and potential_u not in visited_nodes:
                        visited_nodes.add(potential_u)
                        next_layer.add(potential_u)
            current_layer = next_layer
        for u in visited_nodes:
            for v, weight in self.adjacency_counts.get(u, {}).items():
                if v in visited_nodes and weight > 0:
                    g.add_edge(u, v, weight=weight)
        return g

    def compute_node_metrics(self, employee_id: str) -> Tuple[float, float, float, float]:
        matched_nodes = self._resolve_candidate_nodes(employee_id)
        if not matched_nodes:
            return 0.0, 0.0, 0.5, 0.0
        root_node = matched_nodes[0]
        ego_g = self.build_ego_graph(root_node, radius=2)
        if ego_g.number_of_nodes() <= 1:
            return 0.0, 0.0, 0.5, 0.0
        in_degree = float(ego_g.in_degree(root_node, weight="weight"))
        out_degree = float(ego_g.out_degree(root_node, weight="weight"))
        outgoing = set(ego_g.successors(root_node))
        incoming = set(ego_g.predecessors(root_node))
        mutual = len(outgoing.intersection(incoming))
        reciprocity = float((mutual + 1.0) / (len(outgoing) + 2.0))
        betweenness_dict = nx.betweenness_centrality(ego_g, normalized=True, weight="weight")
        betweenness = float(betweenness_dict.get(root_node, 0.0))
        return in_degree, out_degree, reciprocity, betweenness

    def update_context(self, context: EmployeeContext) -> EmployeeContext:
        in_deg, out_deg, recip, betw = self.compute_node_metrics(context.employee_id)
        context.indegree = in_deg
        context.outdegree = out_deg
        context.reciprocity = recip
        context.betweenness = betw
        return context