"""
decision_tree.py
Interpretable decision tree model for tumor classification.

Provides:
  - Train decision tree on extracted features
  - Visualize decision path
  - Extract rules in plain English
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import io
import base64
from typing import Dict, List, Tuple


class DecisionTreeExplainer:
    """
    Train and explain using decision trees.
    Highly interpretable model as baseline comparison.
    """
    
    FEATURE_NAMES = [
        'tumor_area', 'tumor_perimeter', 'circularity',
        'solidity', 'aspect_ratio', 'mean_intensity',
        'std_intensity', 'location_x', 'location_y',
        'bbox_width', 'bbox_height'
    ]
    
    def __init__(self, max_depth=5, use_random_forest=False):
        """
        Args:
            max_depth: Maximum tree depth (smaller = more interpretable)
            use_random_forest: If True, use ensemble of trees
        """
        self.max_depth = max_depth
        self.use_random_forest = use_random_forest
        
        if use_random_forest:
            self.model = RandomForestClassifier(
                n_estimators=10,
                max_depth=max_depth,
                random_state=42
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=42,
                min_samples_split=10,
                min_samples_leaf=5
            )
        
        self.is_trained = False
    
    def train_explainer(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str] = None
    ):
        """
        Train decision tree on tabular features.
        
        Args:
            features: (n_samples, n_features)
            labels: (n_samples,) - binary (0: no tumor, 1: tumor)
            feature_names: Optional custom feature names
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = self.FEATURE_NAMES[:features.shape[1]]
        
        self.model.fit(features, labels)
        self.is_trained = True
        
        # Compute feature importance
        self.feature_importance = {
            name: float(importance)
            for name, importance in zip(
                self.feature_names,
                self.model.feature_importances_
            )
        }
    
    def explain_sample(self, sample_features: np.ndarray) -> Dict:
        """
        Explain a single prediction using the decision tree.
        
        Args:
            sample_features: Feature vector (1D array)
        
        Returns:
            {
                "prediction": int (0 or 1),
                "probability": float,
                "decision_path": [str],
                "feature_splits": dict,
                "tree_visualization": base64_image,
                "rules_applied": [str]
            }
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train_explainer first.")
        
        # Reshape if needed
        if len(sample_features.shape) == 1:
            sample_features = sample_features.reshape(1, -1)
        
        # Predict
        prediction = int(self.model.predict(sample_features)[0])
        probability = float(self.model.predict_proba(sample_features)[0, 1])
        
        # Get decision path
        if self.use_random_forest:
            # For random forest, use first tree
            decision_path, rules = self._explain_single_tree(
                self.model.estimators_[0],
                sample_features[0]
            )
        else:
            decision_path, rules = self._explain_single_tree(
                self.model,
                sample_features[0]
            )
        
        # Feature splits used in decision
        feature_splits = self._get_feature_splits(sample_features[0])
        
        # Visualize tree
        tree_viz = self._visualize_tree()
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "decision_path": decision_path,
            "feature_splits": feature_splits,
            "tree_visualization": tree_viz,
            "rules_applied": rules
        }
    
    def _explain_single_tree(
        self,
        tree_model,
        sample: np.ndarray
    ) -> Tuple[List[str], List[str]]:
        """Extract decision path from tree for a sample."""
        decision_path = []
        rules = []
        
        # Get decision path
        node_indicator = tree_model.decision_path(sample.reshape(1, -1))
        leaf_id = tree_model.apply(sample.reshape(1, -1))
        
        # Get the nodes along the path
        node_index = node_indicator.indices[
            node_indicator.indptr[0]:node_indicator.indptr[1]
        ]
        
        for node_id in node_index:
            # Skip leaf nodes
            if leaf_id[0] == node_id:
                continue
            
            # Get split feature and threshold
            feature_id = tree_model.tree_.feature[node_id]
            threshold = tree_model.tree_.threshold[node_id]
            
            feature_name = self.feature_names[feature_id]
            feature_value = sample[feature_id]
            
            if feature_value <= threshold:
                comparison = f"≤ {threshold:.3f}"
                direction = "left"
            else:
                comparison = f"> {threshold:.3f}"
                direction = "right"
            
            decision = f"{feature_name} {comparison} (value: {feature_value:.3f})"
            decision_path.append(decision)
            
            rule = f"IF {feature_name} {comparison} THEN go {direction}"
            rules.append(rule)
        
        return decision_path, rules
    
    def _get_feature_splits(self, sample: np.ndarray) -> Dict:
        """Get all feature values used in decision."""
        splits = {}
        
        for i, (name, value) in enumerate(zip(self.feature_names, sample)):
            splits[name] = {
                "value": float(value),
                "importance": float(self.feature_importance.get(name, 0))
            }
        
        return splits
    
    def _visualize_tree(self) -> str:
        """Create tree visualization."""
        try:
            if self.use_random_forest:
                # Visualize first tree
                tree_to_plot = self.model.estimators_[0]
            else:
                tree_to_plot = self.model
            
            fig, ax = plt.subplots(figsize=(20, 10))
            fig.patch.set_facecolor('#0a0e1a')
            
            tree.plot_tree(
                tree_to_plot,
                feature_names=self.feature_names,
                class_names=['No Tumor', 'Tumor'],
                filled=True,
                rounded=True,
                fontsize=8,
                ax=ax
            )
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100,
                       facecolor='#0a0e1a', bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error visualizing tree: {e}")
            return ""
    
    def get_rules_as_text(self) -> str:
        """Export tree rules as plain text."""
        if self.use_random_forest:
            tree_model = self.model.estimators_[0]
        else:
            tree_model = self.model
        
        rules = export_text(
            tree_model,
            feature_names=self.feature_names
        )
        
        return rules
    
    def get_feature_importance_plot(self) -> str:
        """Create feature importance bar chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0a0e1a')
            
            # Sort by importance
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            names = [x[0] for x in sorted_features]
            values = [x[1] for x in sorted_features]
            
            y_pos = np.arange(len(names))
            
            ax.barh(y_pos, values, color='#00e5ff', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, color='#e8edf5', fontsize=9)
            ax.set_xlabel('Feature Importance', color='#e8edf5', fontsize=10)
            ax.set_title('Decision Tree Feature Importance',
                        color='#00e5ff', fontsize=12, fontweight='bold')
            
            ax.tick_params(colors='#8899b0')
            ax.spines['bottom'].set_color('#1e2d4a')
            ax.spines['left'].set_color('#1e2d4a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.1, color='#1e2d4a')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100,
                       facecolor='#0a0e1a', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error creating importance plot: {e}")
            return ""


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    print("=" * 70)
    print("  Decision Tree Explainer - Test")
    print("=" * 70)
    print()
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 200
    
    features = np.random.rand(n_samples, 11)
    
    # Create labels based on simple rule
    # If tumor_area > 0.5 OR circularity < 0.3 → tumor
    labels = ((features[:, 0] > 0.5) | (features[:, 2] < 0.3)).astype(int)
    
    print(f"Training data: {n_samples} samples")
    print(f"  Tumor cases: {np.sum(labels)}")
    print(f"  No tumor: {n_samples - np.sum(labels)}")
    print()
    
    # Train explainer
    explainer = DecisionTreeExplainer(max_depth=4)
    explainer.train_explainer(features, labels)
    
    print("✅ Decision tree trained")
    print()
    
    # Test explanation
    test_sample = features[0]
    result = explainer.explain_sample(test_sample)
    
    print("Sample Explanation:")
    print(f"  Prediction: {'Tumor' if result['prediction'] == 1 else 'No Tumor'}")
    print(f"  Probability: {result['probability']:.2%}")
    print()
    
    print("Decision Path:")
    for i, step in enumerate(result['decision_path'], 1):
        print(f"  {i}. {step}")
    print()
    
    print("Rules Applied:")
    for rule in result['rules_applied']:
        print(f"  • {rule}")
    print()
    
    print("Feature Importance:")
    sorted_importance = sorted(
        explainer.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for name, importance in sorted_importance[:5]:
        print(f"  {name}: {importance:.4f}")
    print()
    
    # Get rules as text
    print("Tree Rules (text format):")
    print(explainer.get_rules_as_text())
    print()
    
    print("✅ Test complete")
    
    if result['tree_visualization']:
        print("✅ Tree visualization generated")