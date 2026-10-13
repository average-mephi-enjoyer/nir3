from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Set, Tuple
import networkx as nx
from core.models import CommunicationMessage, EmployeeContext


class DynamicCommunicationGraph:
    """
    Time-decayed directed communication multigraph.
    Constructs localized ego-networks to calculate social isolation and reciprocity drift.
    """

    def __init__(self, decay_window_days: int = 30):
        self.decay_window = timedelta(days=decay_window_days)
        self.edge_stream: Deque[Tuple[datetime, str, str]] = deque()
        self.adjacency_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_communication(self, message: CommunicationMessage) -> None:
        """
        Ingest message into the directed communication graph and prune edges older than decay window.
        """
        sender = message.sender_id.lower().strip()
        ts = message.timestamp
        for recipient in message.receiver_ids:
            target = recipient.lower().strip()
            if sender == target:
                continue
            self.edge_stream.append((ts, sender, target))
            self.adjacency_counts[sender][target] += 1
        self._evict_expired(ts)

    def _evict_expired(self, current_time: datetime) -> None:
        """
        Remove edges older than decay horizon and update adjacency weights.
        """
        cutoff = current_time - self.decay_window
        while self.edge_stream and self.edge_stream[0][0] < cutoff:
            _, src, tgt = self.edge_stream.popleft()
            self.adjacency_counts[src][tgt] -= 1
            if self.adjacency_counts[src][tgt] <= 0:
                del self.adjacency_counts[src][tgt]
            if not self.adjacency_counts[src]:
                del self.adjacency_counts[src]

    def build_ego_graph(self, employee_id: str, radius: int = 2) -> nx.DiGraph:
        """
        Construct localized ego-subgraph for the target node within a bounded radius.
        Prevents global O(V*E) overhead during betweenness calculations.
        """
        norm_id = employee_id.lower().strip()
        g = nx.DiGraph()
        if norm_id not in self.adjacency_counts and not any(norm_id in targets for targets in self.adjacency_counts.values()):
            g.add_node(norm_id)
            return g
        visited_nodes: Set[str] = {norm_id}
        current_layer: Set[str] = {norm_id}
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
        if not g.has_node(norm_id):
            g.add_node(norm_id)
        return g

    def compute_node_metrics(self, employee_id: str) -> Tuple[float, float, float, float]:
        """
        Compute graph topology metrics:
        Returns: (indegree, outdegree, reciprocity, betweenness_centrality).
        """
        norm_id = employee_id.lower().strip()
        ego_g = self.build_ego_graph(norm_id, radius=2)

        if ego_g.number_of_nodes() <= 1:
            return 0.0, 0.0, 0.0, 0.0
        in_degree = float(ego_g.in_degree(norm_id, weight="weight"))
        out_degree = float(ego_g.out_degree(norm_id, weight="weight"))
        outgoing_partners = set(ego_g.successors(norm_id))
        incoming_partners = set(ego_g.predecessors(norm_id))

        if outgoing_partners:
            mutual = outgoing_partners.intersection(incoming_partners)
            reciprocity = float(len(mutual) / len(outgoing_partners))
        else:
            reciprocity = 0.0
        betweenness_dict = nx.betweenness_centrality(ego_g, normalized=True, weight="weight")
        betweenness = float(betweenness_dict.get(norm_id, 0.0))
        return in_degree, out_degree, reciprocity, betweenness

    def update_context(self, context: EmployeeContext) -> EmployeeContext:
        """
        Enrich EmployeeContext with fresh topological graph metrics.
        """
        in_deg, out_deg, recip, betw = self.compute_node_metrics(context.employee_id)
        context.indegree = in_deg
        context.outdegree = out_deg
        context.reciprocity = recip
        context.betweenness = betw
        return context
