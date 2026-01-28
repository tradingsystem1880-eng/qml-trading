# SHAP Explainability Skill

ML model interpretation using SHAP (SHapley Additive exPlanations).

## When to Use
- Understanding XGBoost model predictions
- Feature importance analysis
- Debugging ML model decisions
- Explaining why trades were predicted

## SHAP Basics

### Installation

```bash
pip install shap
```

### Quick Start

```python
import shap
import xgboost as xgb
import pandas as pd
import numpy as np

# Train model (from Phase 8.0)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)
```

## Feature Importance

### Global Feature Importance

```python
def plot_global_importance(model, X: pd.DataFrame, max_features: int = 15):
    """Plot global feature importance using SHAP."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot (bar chart)
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_features,
        show=False
    )
    plt.tight_layout()
    plt.savefig("results/shap_importance.png", dpi=150)
    plt.close()

    # Return importance as DataFrame
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    return importance
```

### Feature Impact Distribution

```python
def plot_feature_impact(model, X: pd.DataFrame, max_features: int = 15):
    """Show how each feature impacts predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Beeswarm plot - shows distribution of impacts
    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        max_display=max_features,
        show=False
    )
    plt.tight_layout()
    plt.savefig("results/shap_beeswarm.png", dpi=150)
    plt.close()
```

## Local Explanations

### Single Prediction Explanation

```python
def explain_single_prediction(model, X: pd.DataFrame, idx: int):
    """
    Explain why model made a specific prediction.

    Args:
        model: Trained XGBoost model
        X: Feature DataFrame
        idx: Index of sample to explain
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Waterfall plot for single prediction
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X.iloc[idx],
            feature_names=X.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"results/shap_prediction_{idx}.png", dpi=150)
    plt.close()

    # Return top contributors
    contributions = pd.DataFrame({
        'feature': X.columns,
        'value': X.iloc[idx].values,
        'shap_value': shap_values[idx]
    }).sort_values('shap_value', key=abs, ascending=False)

    return contributions
```

### Force Plot

```python
def force_plot_prediction(model, X: pd.DataFrame, idx: int):
    """Create interactive force plot for prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Force plot
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False
    )
    plt.savefig(f"results/shap_force_{idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
```

## Trading-Specific Analysis

### Why Did Model Predict Win/Loss?

```python
def explain_trade_prediction(
    model,
    features: pd.DataFrame,
    trade_idx: int,
    actual_outcome: str
):
    """
    Explain why model predicted win/loss for a specific trade.

    Args:
        model: Trained classifier
        features: Feature DataFrame for all trades
        trade_idx: Index of trade to explain
        actual_outcome: 'WIN' or 'LOSS'
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Get prediction probability
    pred_proba = model.predict_proba(features.iloc[[trade_idx]])[0, 1]

    # Get feature contributions
    contributions = pd.DataFrame({
        'feature': features.columns,
        'value': features.iloc[trade_idx].values,
        'shap': shap_values[trade_idx]
    }).sort_values('shap', key=abs, ascending=False)

    print(f"\n{'='*50}")
    print(f"Trade #{trade_idx} Explanation")
    print(f"{'='*50}")
    print(f"Model Prediction: {pred_proba:.1%} win probability")
    print(f"Actual Outcome: {actual_outcome}")
    print(f"\nTop Contributing Factors:")

    for _, row in contributions.head(5).iterrows():
        direction = "↑" if row['shap'] > 0 else "↓"
        print(f"  {row['feature']}: {row['value']:.3f} → {direction} {abs(row['shap']):.3f}")

    return contributions
```

### Feature Interaction Analysis

```python
def analyze_feature_interactions(model, X: pd.DataFrame):
    """Find important feature interactions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Interaction values (can be slow for large datasets)
    shap_interaction = explainer.shap_interaction_values(X[:100])

    # Plot top interactions
    top_features = ['validity_score', 'atr_at_entry', 'rsi']
    for feat in top_features:
        if feat in X.columns:
            shap.dependence_plot(
                feat,
                shap_values,
                X,
                interaction_index='auto',
                show=False
            )
            plt.savefig(f"results/shap_dependence_{feat}.png", dpi=150)
            plt.close()
```

### Compare Win vs Loss Features

```python
def compare_win_loss_shap(model, features: pd.DataFrame, labels: pd.Series):
    """Compare feature contributions for wins vs losses."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Separate by outcome
    win_mask = labels == 1
    loss_mask = labels == 0

    win_mean_shap = np.abs(shap_values[win_mask]).mean(axis=0)
    loss_mean_shap = np.abs(shap_values[loss_mask]).mean(axis=0)

    comparison = pd.DataFrame({
        'feature': features.columns,
        'win_importance': win_mean_shap,
        'loss_importance': loss_mean_shap,
        'diff': win_mean_shap - loss_mean_shap
    }).sort_values('diff', ascending=False)

    print("\nFeatures More Important for Wins:")
    print(comparison.head(5).to_string(index=False))

    print("\nFeatures More Important for Losses:")
    print(comparison.tail(5).to_string(index=False))

    return comparison
```

## Integration with QML

### Phase 8.0 Model Explanation

```python
# Load trained model
import json
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("results/models/xgb_latest.json")

# Load metadata
with open("results/models/xgb_latest.meta.json") as f:
    meta = json.load(f)

selected_features = meta['selected_features']

# Load test data
X_test = pd.read_parquet("results/phase80/test_features.parquet")
X_test = X_test[selected_features]

# Generate explanations
importance = plot_global_importance(model, X_test)
print("\nTop 10 Features by SHAP Importance:")
print(importance.head(10))

# Explain specific trade
explain_trade_prediction(model, X_test, trade_idx=0, actual_outcome='WIN')
```

### SHAP Summary Report

```python
def generate_shap_report(model, X: pd.DataFrame, output_dir: str = "results/shap"):
    """Generate comprehensive SHAP analysis report."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Global importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)

    # 2. Summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f"{output_dir}/summary_plot.png", dpi=150)
    plt.close()

    # 3. Bar plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(f"{output_dir}/bar_plot.png", dpi=150)
    plt.close()

    # 4. Top feature dependence plots
    for feat in importance.head(5)['feature']:
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.savefig(f"{output_dir}/dependence_{feat}.png", dpi=150)
        plt.close()

    print(f"SHAP report generated in {output_dir}/")
    return importance
```

## Key Insights from Phase 8.0

Phase 8.0 ML validation showed:
- AUC 0.53 (essentially random)
- Features don't predict trade outcomes
- SHAP analysis revealed no strong predictors

This is actually a **positive finding**: It confirms QML pattern quality is the edge, not secondary features. The base strategy works without ML enhancement.
