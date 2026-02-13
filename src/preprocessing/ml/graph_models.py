"""
Graph Neural Network for Fraud Detection
Uses PyTorch Geometric to detect fraud rings and network patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader 
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple


class FraudGNN(nn.Module):
    """
    Graph Neural Network for detecting fraud patterns in transaction networks
    
    Architecture:
    - Multiple Graph Convolution layers
    - Attention mechanism
    - Skip connections
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super(FraudGNN, self).__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.dropout = dropout
        
        # Input layer
        if use_attention:
            self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, dropout=dropout)
            self.conv_layers = nn.ModuleList([
                GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
                for _ in range(num_layers - 1)
            ])
        else:
            self.conv1 = SAGEConv(num_node_features, hidden_channels)
            self.conv_layers = nn.ModuleList([
                SAGEConv(hidden_channels, hidden_channels)
                for _ in range(num_layers - 1)
            ])
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * (4 if use_attention else 1))
            for _ in range(num_layers)
        ])
        
        # Output layers
        final_dim = hidden_channels * (4 if use_attention else 1)
        self.fc1 = nn.Linear(final_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector (for batched graphs)
        
        Returns:
            Fraud probability for each node
        """
        
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Additional graph convolution layers with skip connections
        for i, conv in enumerate(self.conv_layers):
            x_residual = x
            x = conv(x, edge_index)
            x = self.batch_norms[i + 1](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Skip connection (if dimensions match)
            if x.shape == x_residual.shape:
                x = x + x_residual
        
        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc3(x)
        
        return torch.sigmoid(x)


class TransactionGraphBuilder:
    """
    Builds transaction graphs for GNN processing
    
    Creates graphs where:
    - Nodes: Users, Merchants, Devices
    - Edges: Transactions, Shared devices, etc.
    """
    
    def __init__(self):
        self.user_mapping = {}
        self.merchant_mapping = {}
        self.device_mapping = {}
        
    def build_graph(
        self,
        transactions_df,
        lookback_days: int = 30
    ) -> Data:
        """
        Build PyTorch Geometric graph from transactions
        
        Args:
            transactions_df: Transaction data
            lookback_days: How many days of history to include
        
        Returns:
            PyTorch Geometric Data object
        """
        
        # Create NetworkX graph first
        G = nx.Graph()
        
        # Add nodes
        users = transactions_df['user_id'].unique()
        merchants = transactions_df['merchant_id'].unique()
        
        # Node indexing
        node_idx = 0
        for user in users:
            self.user_mapping[user] = node_idx
            G.add_node(node_idx, node_type='user')
            node_idx += 1
        
        for merchant in merchants:
            self.merchant_mapping[merchant] = node_idx
            G.add_node(node_idx, node_type='merchant')
            node_idx += 1
        
        # Add edges (transactions)
        edge_list = []
        edge_features = []
        node_labels = {}
        
        for _, txn in transactions_df.iterrows():
            user_idx = self.user_mapping[txn['user_id']]
            merchant_idx = self.merchant_mapping[txn['merchant_id']]
            
            edge_list.append([user_idx, merchant_idx])
            edge_features.append([
                txn['amount'],
                txn.get('hour', 0),
                txn.get('is_weekend', 0)
            ])
            
            # Store fraud labels
            if 'is_fraud' in txn:
                node_labels[user_idx] = txn['is_fraud']
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create node features (for now, simple degree-based features)
        num_nodes = len(G.nodes())
        node_features = self._create_node_features(G, transactions_df)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Labels
        y = torch.zeros(num_nodes, dtype=torch.float)
        for node_idx, label in node_labels.items():
            y[node_idx] = label
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        
        return data
    
    def _create_node_features(self, G: nx.Graph, transactions_df) -> np.ndarray:
        """Create node feature matrix"""
        
        num_nodes = len(G.nodes())
        features = []
        
        for node in range(num_nodes):
            # Basic graph features
            degree = G.degree(node) if node in G else 0
            
            # Create feature vector
            node_feat = [
                degree,
                np.log1p(degree),
                # Add more features as needed
            ]
            
            features.append(node_feat)
        
        return np.array(features)
    
    def detect_fraud_rings(
        self,
        transactions_df,
        model: FraudGNN,
        threshold: float = 0.7
    ) -> List[List[int]]:
        """
        Detect fraud rings (groups of connected fraudulent accounts)
        
        Args:
            transactions_df: Transaction data
            model: Trained GNN model
            threshold: Fraud probability threshold
        
        Returns:
            List of fraud rings (each ring is a list of user IDs)
        """
        
        # Build graph
        graph_data = self.build_graph(transactions_df)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model(graph_data.x, graph_data.edge_index)
        
        # Find suspicious nodes
        suspicious_nodes = (predictions.squeeze() > threshold).nonzero().squeeze().tolist()
        
        # Build NetworkX graph of suspicious connections
        G_suspicious = nx.Graph()
        
        edge_index = graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in suspicious_nodes and dst in suspicious_nodes:
                G_suspicious.add_edge(src, dst)
        
        # Find connected components (fraud rings)
        fraud_rings = list(nx.connected_components(G_suspicious))
        
        return [list(ring) for ring in fraud_rings if len(ring) > 1]


class HybridFraudDetector:
    """
    Combines GNN with traditional ML for best performance
    
    Ensemble approach:
    - GNN for network patterns
    - XGBoost for transaction features
    - Weighted voting
    """
    
    def __init__(
        self,
        gnn_model: FraudGNN,
        xgboost_model,
        gnn_weight: float = 0.6
    ):
        self.gnn_model = gnn_model
        self.xgboost_model = xgboost_model
        self.gnn_weight = gnn_weight
        self.xgb_weight = 1 - gnn_weight
        
    def predict(
        self,
        graph_data: Data,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Ensemble prediction
        
        Args:
            graph_data: PyTorch Geometric graph
            features: Traditional features for XGBoost
        
        Returns:
            Combined fraud probabilities
        """
        
        # GNN predictions
        self.gnn_model.eval()
        with torch.no_grad():
            gnn_pred = self.gnn_model(
                graph_data.x,
                graph_data.edge_index
            ).numpy().flatten()
        
        # XGBoost predictions
        xgb_pred = self.xgboost_model.predict_proba(features)[:, 1]
        
        # Weighted ensemble
        ensemble_pred = (
            self.gnn_weight * gnn_pred +
            self.xgb_weight * xgb_pred
        )
        
        return ensemble_pred
    
    def explain_prediction(
        self,
        transaction_idx: int,
        graph_data: Data,
        features: np.ndarray
    ) -> Dict:
        """
        Explain why a transaction was flagged as fraud
        
        Returns:
            Dictionary with explanation components
        """
        
        gnn_score = self.gnn_model(
            graph_data.x,
            graph_data.edge_index
        )[transaction_idx].item()
        
        xgb_score = self.xgboost_model.predict_proba(
            features[transaction_idx:transaction_idx+1]
        )[0, 1]
        
        final_score = (
            self.gnn_weight * gnn_score +
            self.xgb_weight * xgb_score
        )
        
        return {
            'final_score': final_score,
            'gnn_contribution': gnn_score * self.gnn_weight,
            'xgb_contribution': xgb_score * self.xgb_weight,
            'network_risk': gnn_score,
            'feature_risk': xgb_score,
            'ensemble_method': 'weighted_voting'
        }


if __name__ == "__main__":
    print("Graph Neural Network Module - Ready")
    print("Detects fraud rings and network-based fraud patterns")