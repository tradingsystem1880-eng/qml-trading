"""
ML Training Page - Feature engineering and model training interface.

Phase 7 implementation with:
- Feature analysis and data quality checks
- Purged cross-validation (no data leakage)
- Imbalanced class handling
- SHAP-based feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

from theme import ARCTIC_PRO, TYPOGRAPHY

# Try to import ML modules
try:
    from src.ml import MLTrainingPipeline, TrainingResult, PurgedKFold, WalkForwardCV
    from src.data.sqlite_manager import SQLiteManager, get_db
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)


def render_ml_training_page() -> None:
    """Render the ML training and evaluation page."""

    # Page header
    html = '<div class="panel">'
    html += '<div class="panel-header">Neuro-Lab (ML Training)</div>'
    html += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Train models with purged cross-validation to predict trade outcomes.</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    if not ML_AVAILABLE:
        _render_import_error()
        return

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Analysis",
        "Train Model",
        "Results",
        "Feature Importance"
    ])

    try:
        db = get_db()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return

    with tab1:
        _render_feature_analysis(db)

    with tab2:
        _render_training_interface(db)

    with tab3:
        _render_results()

    with tab4:
        _render_feature_importance()


def _render_import_error() -> None:
    """Render import error message."""
    error_html = '<div class="panel">'
    error_html += f'<div style="color: {ARCTIC_PRO["danger"]}; padding: 1rem;">'
    error_html += '<strong>ML module not available</strong><br>'
    error_html += f'<span style="color: {ARCTIC_PRO["text_muted"]};">Import error: {ML_IMPORT_ERROR}</span>'
    error_html += '</div>'
    error_html += '</div>'
    st.markdown(error_html, unsafe_allow_html=True)


def _render_feature_analysis(db: SQLiteManager) -> None:
    """Render feature analysis section."""

    analysis_panel = '<div class="panel">'
    analysis_panel += '<div class="panel-header">Feature Analysis</div>'
    analysis_panel += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Analyze feature distributions and data quality.</p>'
    analysis_panel += '</div>'
    st.markdown(analysis_panel, unsafe_allow_html=True)

    # Try to load features from database
    try:
        features_df = db.get_features_for_training(with_outcomes_only=True)

        if not features_df:
            st.warning("No features calculated yet.")
            st.info(
                "To generate features:\n"
                "1. Go to **Backtest** tab and run a backtest\n"
                "2. Features will be calculated and stored automatically"
            )
            return

        features_df = pd.DataFrame(features_df)

        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            _render_metric_card("Total Samples", str(len(features_df)))
        with col2:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            _render_metric_card("Features", str(len(numeric_cols)))
        with col3:
            if 'outcome' in features_df.columns:
                win_rate = (features_df['outcome'] == 'WIN').mean()
                _render_metric_card("Win Rate", f"{win_rate:.1%}")
            else:
                _render_metric_card("Win Rate", "N/A")
        with col4:
            missing_pct = features_df.isnull().sum().sum() / features_df.size * 100
            _render_metric_card("Missing Data", f"{missing_pct:.1f}%")

        # Data quality warnings
        if missing_pct > 5:
            st.warning(f"High missing data ({missing_pct:.1f}%). Consider imputation or dropping columns.")

        # Feature statistics
        st.markdown("### Feature Statistics")

        # Get numeric columns only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            stats_df = features_df[numeric_cols].describe().T
            stats_df['missing'] = features_df[numeric_cols].isnull().sum()
            stats_df['missing_pct'] = (stats_df['missing'] / len(features_df) * 100).round(1)

            st.dataframe(stats_df, use_container_width=True)

        # Correlation matrix
        if len(numeric_cols) > 1:
            st.markdown("### Feature Correlations")

            valid_cols = [c for c in numeric_cols if features_df[c].notna().any()]
            if len(valid_cols) > 1:
                corr_matrix = features_df[valid_cols].corr()

                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    zmin=-1, zmax=1
                )
                fig.update_layout(
                    plot_bgcolor=ARCTIC_PRO['bg_secondary'],
                    paper_bgcolor=ARCTIC_PRO['bg_card'],
                    font_color=ARCTIC_PRO['text_secondary'],
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Highly correlated features warning
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.9:
                            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

                if high_corr:
                    st.warning(f"{len(high_corr)} highly correlated feature pairs (>0.9). Consider removing redundant features.")

    except Exception as e:
        st.error(f"Error loading features: {e}")
        st.info("Check that the database exists and contains feature data.")


def _render_training_interface(db: SQLiteManager) -> None:
    """Render model training section."""

    train_panel = '<div class="panel">'
    train_panel += '<div class="panel-header">Train Model</div>'
    train_panel += f'<p style="color: {ARCTIC_PRO["text_muted"]};">Configure and train ML model with purged cross-validation.</p>'
    train_panel += '</div>'
    st.markdown(train_panel, unsafe_allow_html=True)

    # Model configuration
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Model Type",
            ['xgboost', 'lightgbm'],
            help="XGBoost and LightGBM are best for tabular financial data"
        )

        cv_method = st.selectbox(
            "Cross-Validation Method",
            ['purged', 'walk_forward'],
            help="Purged CV prevents lookahead bias. Walk-forward simulates real trading."
        )

        n_folds = st.slider("Number of Folds", 3, 10, 5)

        handle_imbalanced = st.checkbox(
            "Auto handle class imbalance",
            value=True,
            help="Automatically adjusts class weights for win/loss imbalance"
        )

    with col2:
        n_estimators = st.slider("Number of Trees", 50, 500, 100, step=50)
        max_depth = st.slider("Max Depth", 3, 10, 5)
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
        early_stopping = st.slider(
            "Early Stopping Rounds",
            0, 50, 10,
            help="0 = no early stopping"
        )

    # Load features
    try:
        features_data = db.get_features_for_training(with_outcomes_only=True)

        if not features_data:
            st.warning("No features available for training.")
            st.info("Run feature calculation from the **Backtest** tab first.")
            return

        features_df = pd.DataFrame(features_data)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove non-feature columns
        exclude_cols = ['pattern_id', 'r_multiple', 'calculation_time', 'id']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        st.info(f"Available features: {len(feature_cols)}")

        # Feature selection
        selected_features = st.multiselect(
            "Select Features",
            feature_cols,
            default=feature_cols[:min(10, len(feature_cols))]
        )

        if len(selected_features) < 3:
            st.warning("Select at least 3 features for meaningful training")
            return

        # Show class balance
        if 'outcome' in features_df.columns:
            win_rate = (features_df['outcome'] == 'WIN').mean()
            st.info(f"Class balance: {win_rate:.1%} wins / {1 - win_rate:.1%} losses")

            if win_rate < 0.2 or win_rate > 0.8:
                st.warning("Highly imbalanced classes. 'Auto handle class imbalance' is recommended.")

        # Train button
        st.markdown("---")

        if st.button("Train Model", type="primary", use_container_width=True, key="train_ml"):
            with st.spinner("Training model with purged cross-validation..."):
                try:
                    # Prepare data
                    X = features_df[selected_features].copy()

                    # Handle missing values
                    missing_count = X.isnull().sum().sum()
                    if missing_count > 0:
                        st.info(f"Filling {missing_count} missing values with 0")
                        X = X.fillna(0)

                    # Replace inf with large values
                    X = X.replace([np.inf, -np.inf], [1e10, -1e10])

                    # Create target (WIN=1, LOSS=0)
                    if 'outcome' in features_df.columns:
                        y = (features_df['outcome'] == 'WIN').astype(int)
                    elif 'r_multiple' in features_df.columns:
                        y = (features_df['r_multiple'] > 0).astype(int)
                    else:
                        st.error("No target variable found (need 'outcome' or 'r_multiple' column)")
                        return

                    # Train
                    pipeline = MLTrainingPipeline(
                        model_type=model_type,
                        cv_method=cv_method,
                        n_folds=n_folds,
                        handle_imbalanced=handle_imbalanced,
                        early_stopping_rounds=early_stopping
                    )

                    result = pipeline.train(
                        X, y,
                        model_params={
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate
                        }
                    )

                    # Save result to session state
                    st.session_state['ml_result'] = result
                    st.session_state['ml_pipeline'] = pipeline

                    # Save model
                    pipeline.save('models/qml_classifier.joblib')

                    st.success(
                        f"Model trained successfully!\n\n"
                        f"AUC-ROC: **{result.metrics.roc_auc:.3f}**\n\n"
                        f"F1 Score: **{result.metrics.f1:.3f}**\n\n"
                        f"Training time: {result.training_time_seconds:.1f}s"
                    )

                except ValueError as e:
                    st.error(f"Data validation error: {e}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.info("Check that your features don't contain NaN or infinite values.")

    except Exception as e:
        st.error(f"Error: {e}")


def _render_results() -> None:
    """Render training results section."""

    results_panel = '<div class="panel">'
    results_panel += '<div class="panel-header">Training Results</div>'
    results_panel += '</div>'
    st.markdown(results_panel, unsafe_allow_html=True)

    if 'ml_result' not in st.session_state:
        st.info("No model trained yet. Go to **Train Model** tab.")
        return

    result: TrainingResult = st.session_state['ml_result']

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _render_metric_card("AUC-ROC", f"{result.metrics.roc_auc:.3f}")
    with col2:
        _render_metric_card("Precision", f"{result.metrics.precision:.3f}")
    with col3:
        _render_metric_card("Recall", f"{result.metrics.recall:.3f}")
    with col4:
        _render_metric_card("F1 Score", f"{result.metrics.f1:.3f}")

    # Training info
    st.markdown("### Training Info")

    col1, col2 = st.columns(2)

    with col1:
        info_data = {
            'Parameter': ['Model Type', 'CV Method', 'Folds', 'Features', 'Training Samples', 'Training Time'],
            'Value': [
                result.model_type.upper(),
                result.cv_method.replace('_', ' ').title(),
                str(result.n_folds),
                str(result.n_features),
                str(result.n_samples_train),
                f"{result.training_time_seconds:.1f}s"
            ]
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

    with col2:
        st.write("**Class Balance Handling:**")
        st.write(f"- Positive class ratio: {result.class_ratio:.1%}")
        st.write(f"- Scale pos weight: {result.scale_pos_weight:.2f}")
        st.write(f"- Model version: {result.model_version}")

    # Fold-by-fold metrics
    if result.metrics.fold_metrics:
        st.markdown("### Fold-by-Fold Performance")

        fold_df = pd.DataFrame(result.metrics.fold_metrics)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {f['fold']}" for f in result.metrics.fold_metrics],
            y=fold_df['roc_auc'],
            name='AUC-ROC',
            marker_color=ARCTIC_PRO['accent']
        ))

        # Add mean line
        mean_auc = fold_df['roc_auc'].mean()
        fig.add_hline(y=mean_auc, line_dash="dash", line_color=ARCTIC_PRO['success'],
                      annotation_text=f"Mean: {mean_auc:.3f}")

        fig.update_layout(
            title="AUC by Fold (Purged CV ensures no data leakage)",
            plot_bgcolor=ARCTIC_PRO['bg_secondary'],
            paper_bgcolor=ARCTIC_PRO['bg_card'],
            font_color=ARCTIC_PRO['text_secondary'],
            yaxis_title="AUC-ROC"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fold details table
        with st.expander("Fold Details"):
            st.dataframe(fold_df, use_container_width=True)

    # Confusion matrix
    st.markdown("### Confusion Matrix")

    cm_data = [
        [result.metrics.true_negatives, result.metrics.false_positives],
        [result.metrics.false_negatives, result.metrics.true_positives]
    ]

    fig = px.imshow(
        cm_data,
        labels=dict(x="Predicted", y="Actual"),
        x=['Predict Loss', 'Predict Win'],
        y=['Actual Loss', 'Actual Win'],
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(
        plot_bgcolor=ARCTIC_PRO['bg_secondary'],
        paper_bgcolor=ARCTIC_PRO['bg_card'],
        font_color=ARCTIC_PRO['text_secondary']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.write(f"""
    **Interpretation:**
    - Model correctly identifies **{result.metrics.true_positives}** wins (True Positives)
    - Model correctly identifies **{result.metrics.true_negatives}** losses (True Negatives)
    - Model incorrectly predicts win when actually loss: **{result.metrics.false_positives}** (False Positives)
    - Model incorrectly predicts loss when actually win: **{result.metrics.false_negatives}** (False Negatives)
    """)


def _render_feature_importance() -> None:
    """Render feature importance section."""

    importance_panel = '<div class="panel">'
    importance_panel += '<div class="panel-header">Feature Importance</div>'
    importance_panel += '</div>'
    st.markdown(importance_panel, unsafe_allow_html=True)

    if 'ml_result' not in st.session_state:
        st.info("No model trained yet. Go to **Train Model** tab.")
        return

    result: TrainingResult = st.session_state['ml_result']

    if not result.feature_importance:
        st.warning("No feature importance data available.")
        return

    # Feature importance chart
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in result.feature_importance.items()
    ]).sort_values('importance', ascending=True).tail(20)  # Top 20

    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Top 20 Feature Importances (SHAP-based if available)",
        color='importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        plot_bgcolor=ARCTIC_PRO['bg_secondary'],
        paper_bgcolor=ARCTIC_PRO['bg_card'],
        font_color=ARCTIC_PRO['text_secondary'],
        height=600,
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance interpretation
    top_3 = list(result.feature_importance.keys())[:3]
    if len(top_3) >= 3:
        st.write(f"""
        **Top 3 Most Important Features:**
        1. **{top_3[0]}** - Most predictive of win/loss outcome
        2. **{top_3[1]}** - Second most predictive
        3. **{top_3[2]}** - Third most predictive

        *Higher importance = more influence on model predictions*
        """)

    # Feature importance table
    st.markdown("### All Feature Importances")

    all_importance = pd.DataFrame([
        {'Feature': k, 'Importance': v, 'Rank': i + 1}
        for i, (k, v) in enumerate(result.feature_importance.items())
    ])

    st.dataframe(all_importance, use_container_width=True, hide_index=True)

    # Download button
    csv = all_importance.to_csv(index=False)
    st.download_button(
        "Download Feature Importance CSV",
        csv,
        "feature_importance.csv",
        "text/csv"
    )


def _render_metric_card(label: str, value: str) -> None:
    """Render a simple metric card."""
    html = '<div class="metric-card">'
    html += f'<div class="metric-label">{label}</div>'
    html += f'<div class="metric-value">{value}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
