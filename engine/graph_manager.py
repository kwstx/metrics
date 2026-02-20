from typing import List, Optional, Dict, Any, Set
from sqlalchemy.orm import Session
from models.impact_graph import ImpactNode, ImpactEdge
import networkx as nx

class GraphManager:
    """
    Manages incremental updates, traversal queries, and causal path computations 
    for the Impact Graph.
    """
    def __init__(self, session: Session):
        self.session = session

    def add_node(self, 
                 outcome_type: str, 
                 domain_context: Dict[str, Any], 
                 magnitude: float = 0.0, 
                 uncertainty: Optional[Dict[str, Any]] = None) -> ImpactNode:
        node = ImpactNode(
            outcome_type=outcome_type,
            domain_context=domain_context,
            magnitude=magnitude,
            uncertainty_metadata=uncertainty
        )
        self.session.add(node)
        self.session.commit()
        return node

    def add_causal_edge(self, 
                         source_id: str, 
                         target_id: str, 
                         weight: float, 
                         confidence: float = 1.0, 
                         delay: float = 0.0) -> ImpactEdge:
        edge = ImpactEdge(
            source_node_id=source_id,
            target_node_id=target_id,
            causal_weight=weight,
            confidence_score=confidence,
            propagation_delay=delay
        )
        self.session.add(edge)
        self.session.commit()
        return edge

    def get_causal_paths(self, start_node_id: str, end_node_id: str) -> List[List[ImpactNode]]:
        """
        Computes all causal paths between two nodes.
        Uses networkx for graph traversal logic.
        """
        G = self._build_nx_graph()
        try:
            paths = list(nx.all_simple_paths(G, source=start_node_id, target=end_node_id))
            # Convert node IDs back to ImpactNode objects
            results = []
            for path in paths:
                nodes = self.session.query(ImpactNode).filter(ImpactNode.id.in_(path)).all()
                # Re-sort nodes to match path order
                node_map = {n.id: n for n in nodes}
                results.append([node_map[node_id] for node_id in path])
            return results
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def compute_cumulative_impact(self, node_id: str) -> float:
        """
        Computes the total downstream impact starting from a node.
        Propagates magnitude through causal weights.
        """
        G = self._build_nx_graph()
        if node_id not in G:
            return 0.0

        total_impact = 0.0
        start_node = self.session.get(ImpactNode, node_id)
        if not start_node:
            return 0.0
            
        # simple BFS/DFS traversal to sum up (magnitude * path_weights)
        # Note: This is an initial implementation. More complex causal math might be needed.
        visited = {node_id: start_node.magnitude}
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            current_val = visited[current_id]
            
            for neighbor_id in G.successors(current_id):
                edge_data = G.get_edge_data(current_id, neighbor_id)
                weight = edge_data['weight']
                confidence = edge_data['confidence']
                
                impact_contribution = current_val * weight * confidence
                
                if neighbor_id not in visited:
                    visited[neighbor_id] = 0.0
                    queue.append(neighbor_id)
                
                visited[neighbor_id] += impact_contribution
                total_impact += impact_contribution

        return total_impact

    def _build_nx_graph(self, disabled_source_node_ids: Optional[Set[str]] = None) -> nx.DiGraph:
        """
        Internal helper to build a NetworkX directed graph from the database.
        Allows for efficient graph algorithms.
        """
        G = nx.DiGraph()
        disabled_source_node_ids = disabled_source_node_ids or set()
        nodes = self.session.query(ImpactNode).all()
        edges = self.session.query(ImpactEdge).all()
        
        for n in nodes:
            G.add_node(n.id, magnitude=n.magnitude, type=n.outcome_type)
        
        for e in edges:
            if e.source_node_id in disabled_source_node_ids:
                continue
            G.add_edge(e.source_node_id, e.target_node_id, 
                       weight=e.causal_weight, 
                       confidence=e.confidence_score,
                       delay=e.propagation_delay)
        return G
