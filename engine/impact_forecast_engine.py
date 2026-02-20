from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import ImpactProjection, AgentReliability, DomainMultiplier
from .graph_manager import GraphManager
import networkx as nx
from datetime import datetime

class ImpactForecastEngine:
    """
    Transforms agent actions into probabilistic downstream impact projections.
    Uses causal graph traversal with reliability and domain scaling.
    """
    def __init__(self, session: Session):
        self.session = session
        self.graph_manager = GraphManager(session)

    def forecast_action(self, action_node_id: str, time_horizon: float = 30.0) -> ImpactProjection:
        """
        Takes an action node ID and generates a full ImpactProjection.
        """
        # 1. Fetch action node
        action_node = self.session.query(ImpactNode).filter(ImpactNode.id == action_node_id).first()
        if not action_node:
            raise ValueError(f"Action node {action_node_id} not found")

        # 2. Extract agent and domain context
        agent_id = action_node.domain_context.get("agent_id") or action_node.domain_context.get("agent")
        domain = action_node.domain_context.get("domain")

        # 3. Fetch reliability and multipliers
        reliability = self._get_agent_reliability(agent_id)
        multiplier = self._get_domain_multiplier(domain)

        # 4. Propagate influence across the Impact Graph
        impact_vector = self._compute_projected_vector(action_node_id, reliability, multiplier)

        # 5. Calculate uncertainty and confidence intervals
        base_uncertainty = action_node.uncertainty_metadata or {"std_dev": 0.1, "confidence": 0.9}
        uncertainty_bounds = self._calculate_uncertainty(impact_vector, base_uncertainty, reliability)

        # 6. Gather dependency references
        dependencies = self._get_dependencies(action_node_id)

        # 7. Persist Projection
        projection = ImpactProjection(
            source_node_id=action_node.id,
            predicted_impact_vector=impact_vector,
            uncertainty_bounds=uncertainty_bounds,
            time_horizon=time_horizon,
            dependency_references=dependencies,
            confidence_interval=uncertainty_bounds # Used for recalibration later
        )

        self.session.add(projection)
        self.session.commit()
        return projection

    def _get_agent_reliability(self, agent_id: Optional[str]) -> float:
        if not agent_id:
            return 1.0
        record = self.session.query(AgentReliability).filter(AgentReliability.agent_id == agent_id).first()
        return record.reliability_coefficient if record else 1.0

    def _get_domain_multiplier(self, domain_name: Optional[str]) -> float:
        if not domain_name:
            return 1.0
        record = self.session.query(DomainMultiplier).filter(DomainMultiplier.domain_name == domain_name).first()
        return record.multiplier if record else 1.0

    def _compute_projected_vector(self, node_id: str, reliability: float, multiplier: float) -> Dict[str, float]:
        """
        Propagates influence through connected graph nodes using weighted causal traversal.
        Groups impact by outcome_type to form a multi-dimensional vector.
        """
        G = self.graph_manager._build_nx_graph()
        if node_id not in G:
            return {}

        # Use topological sort if DAG, otherwise BFS with accumulation
        # Causal graphs are generally DAGs.
        is_dag = nx.is_directed_acyclic_graph(G)
        
        # visited stores the relative influence (magnitude multiplier) at each node
        influence_map = {node_id: 1.0}
        vector = {}

        if is_dag:
            # Process nodes in topological order starting from node_id
            nodes_to_process = list(nx.topological_sort(G))
            # Filter to only include descendants of node_id (and node_id itself)
            descendants = nx.descendants(G, node_id)
            descendants.add(node_id)
            
            for n_id in nodes_to_process:
                if n_id not in descendants:
                    continue
                
                inf = influence_map.get(n_id, 0.0)
                if inf <= 0:
                    continue
                
                # Update vector
                node = self.session.get(ImpactNode, n_id)
                if node:
                    v_type = node.outcome_type
                    vector[v_type] = vector.get(v_type, 0.0) + (inf * node.magnitude * multiplier * reliability)
                
                # Propagate to successors
                for neighbor_id in G.successors(n_id):
                    edge_data = G.get_edge_data(n_id, neighbor_id)
                    weight = edge_data['weight']
                    conf = edge_data['confidence']
                    
                    added_influence = inf * weight * conf
                    influence_map[neighbor_id] = influence_map.get(neighbor_id, 0.0) + added_influence
        else:
            # Fallback for graphs with cycles - BFS with limited depth or simple traversal
            # For causal impact, cycles usually represent feedback loops
            queue = [node_id]
            processed = set()
            
            while queue:
                curr_id = queue.pop(0)
                inf = influence_map[curr_id]
                
                node = self.session.get(ImpactNode, curr_id)
                if node:
                    v_type = node.outcome_type
                    vector[v_type] = vector.get(v_type, 0.0) + (inf * node.magnitude * multiplier * reliability)
                
                for neighbor_id in G.successors(curr_id):
                    edge_data = G.get_edge_data(curr_id, neighbor_id)
                    weight = edge_data['weight']
                    conf = edge_data['confidence']
                    
                    influence_map[neighbor_id] = influence_map.get(neighbor_id, 0.0) + (inf * weight * conf)
                    if neighbor_id not in processed:
                        queue.append(neighbor_id)
                        processed.add(neighbor_id)
        
        return vector

    def _calculate_uncertainty(self, vector: Dict[str, float], base_uncertainty: Dict[str, Any], reliability: float) -> Dict[str, Any]:
        """
        Applies a basic uncertainty bounds model based on agent reliability and base metrics.
        """
        # Heuristic: error increases as reliability decreases
        rel_factor = (1.0 - reliability)
        base_std = base_uncertainty.get("std_dev", 0.1)
        
        total_error_coeff = rel_factor + base_std
        
        bounds = {}
        for metric, value in vector.items():
            margin = value * total_error_coeff
            bounds[metric] = {
                "min": value - margin,
                "max": value + margin,
                "confidence_interval": [value - margin, value + margin],
                "expected_value": value
            }
        return bounds

    def _get_dependencies(self, node_id: str) -> List[str]:
        """
        Returns a list of node IDs that form the causal dependency chain.
        """
        G = self.graph_manager._build_nx_graph()
        if node_id not in G:
            return []
        # All nodes reachable from the source
        return list(nx.descendants(G, node_id))
