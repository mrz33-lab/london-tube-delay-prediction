"""
SHAP explainability pipeline.

Generates global feature importance (beeswarm, bar) and local waterfall
explanations from the trained model.
"""

import hashlib
import logging
from pathlib import Path
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config, get_config, RANDOM_SEED
from utils import setup_logging, get_latest_run_id, set_random_seeds, format_duration
from features import load_feature_metadata
from train import NaiveBaselineModel  # noqa: F401 — needed for joblib unpickling

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - explainability will be limited")


logger = logging.getLogger(__name__)


def load_artifacts(artifact_dir: Path) -> dict:
    """Load models, predictions, and feature data from a run directory."""
    artifacts = {}

    for model_name in ['naive', 'ridge', 'best']:
        model_path = artifact_dir / f'{model_name}_model.pkl'
        if model_path.exists():
            artifacts[f'{model_name}_model'] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} model")

    artifacts['feature_metadata'] = load_feature_metadata(artifact_dir)

    test_pred_path = artifact_dir / 'test_predictions.csv'
    if test_pred_path.exists():
        artifacts['test_predictions'] = pd.read_csv(test_pred_path)

    x_train_path = artifact_dir / 'X_train.parquet'
    x_test_path = artifact_dir / 'X_test.parquet'
    if x_train_path.exists():
        artifacts['X_train'] = pd.read_parquet(x_train_path)
    if x_test_path.exists():
        artifacts['X_test'] = pd.read_parquet(x_test_path)

    best_model_name_path = artifact_dir / 'best_model_name.txt'
    if best_model_name_path.exists():
        with open(best_model_name_path, 'r') as f:
            artifacts['best_model_name'] = f.read().strip()

    return artifacts


def _shap_cache_key(artifact_dir: Path, X_test: pd.DataFrame) -> str:
    """MD5 of (model file mtime, X_test data hash) — cheap to compute."""
    model_path = artifact_dir / 'best_model.pkl'
    mtime = str(model_path.stat().st_mtime) if model_path.exists() else 'missing'
    data_hash = str(pd.util.hash_pandas_object(X_test).sum())
    return hashlib.md5((mtime + data_hash).encode()).hexdigest()


def load_shap_cache(artifact_dir: Path, X_test: pd.DataFrame):
    """Return (shap_values, feature_names) from cache, or (None, None) on miss."""
    cache_path = artifact_dir / 'shap_cache.pkl'
    if not cache_path.exists():
        return None, None
    try:
        cache = joblib.load(cache_path)
        if cache.get('key') == _shap_cache_key(artifact_dir, X_test):
            logger.info("SHAP cache hit — skipping recomputation")
            return cache['shap_values'], cache['feature_names']
        logger.info("SHAP cache key mismatch — recomputing")
    except Exception as exc:
        logger.warning("SHAP cache unreadable (%s) — recomputing", exc)
    return None, None


def save_shap_cache(artifact_dir: Path, X_test: pd.DataFrame,
                    shap_values, feature_names) -> None:
    """Persist SHAP values so the next explain run can skip recomputation."""
    cache_path = artifact_dir / 'shap_cache.pkl'
    joblib.dump(
        {'key': _shap_cache_key(artifact_dir, X_test),
         'shap_values': shap_values,
         'feature_names': feature_names},
        cache_path,
    )
    logger.info("Saved SHAP cache to %s", cache_path)


def create_shap_explainer(model, X_background, config):
    """Create SHAP explainer, handling Pipeline vs raw estimator."""
    if not SHAP_AVAILABLE:
        return None

    try:
        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                X_background_transformed = preprocessor.transform(X_background)
            else:
                X_background_transformed = X_background

            regressor = model.named_steps.get('regressor')

            if len(X_background_transformed) > config.explainability.shap_background_size:
                rng = np.random.RandomState(RANDOM_SEED)
                indices = rng.choice(
                    len(X_background_transformed),
                    size=config.explainability.shap_background_size,
                    replace=False
                )
                X_background_transformed = X_background_transformed[indices]

            explainer = shap.Explainer(regressor, X_background_transformed)
        else:
            explainer = shap.Explainer(model, X_background)

        logger.info("Created SHAP explainer")
        return explainer

    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        return None


def compute_shap_values(explainer, model, X, config, max_samples=None):
    if explainer is None:
        return None

    try:
        if max_samples and len(X) > max_samples:
            rng = np.random.RandomState(RANDOM_SEED)
            indices = rng.choice(len(X), size=max_samples, replace=False)
            X_sample = X.iloc[indices]
        else:
            X_sample = X

        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessor')
            X_transformed = preprocessor.transform(X_sample) if preprocessor else X_sample
        else:
            X_transformed = X_sample

        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        try:
            shap_values = explainer(X_transformed, check_additivity=False)
        except TypeError:
            # LinearExplainer does not support check_additivity
            shap_values = explainer(X_transformed)
        logger.info("SHAP values computed")
        return shap_values

    except Exception as e:
        logger.error(f"Failed to compute SHAP values: {e}")
        return None


def create_shap_plots(shap_values, feature_names, output_dir, config):
    if shap_values is None:
        return

    try:
        # beeswarm
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=config.explainability.top_n_features, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()

        # bar
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=config.explainability.top_n_features, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close()

        # waterfall for individual predictions
        n_waterfall = min(config.explainability.n_local_examples, len(shap_values))
        waterfall_saved = 0
        for i in range(n_waterfall):
            try:
                exp_i = shap_values[i]
                exp_i.feature_names = feature_names
                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(exp_i, max_display=config.explainability.top_n_features, show=False)
                plt.savefig(output_dir / f'shap_waterfall_example_{i}.png', dpi=72)
                plt.close()
                waterfall_saved += 1
            except Exception as we:
                logger.warning(f"Waterfall plot {i} skipped: {we}", exc_info=True)
                plt.close()

        logger.info(f"Created {waterfall_saved} waterfall plots")

    except Exception as e:
        logger.error(f"Failed to create SHAP plots: {e}", exc_info=True)


def generate_text_explanations(shap_values, feature_names, X, y_pred, config):
    """Turn SHAP values into plain-English summaries."""
    if shap_values is None:
        return ["SHAP values not available"]

    explanations = []
    try:
        shap_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
        n_examples = min(config.explainability.n_local_examples, len(shap_array))

        for i in range(n_examples):
            shap_vals = shap_array[i]
            top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]

            explanation = f"Prediction #{i+1}: {y_pred[i]:.1f} minutes delay\n"
            explanation += "Main factors:\n"

            for idx in top_indices:
                feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                val = shap_vals[idx]
                direction = "increases" if val > 0 else "decreases"
                explanation += f"  - {feat}: {direction} delay by {abs(val):.2f} min\n"

            explanations.append(explanation)

    except Exception as e:
        logger.error(f"Text explanation failed: {e}")
        explanations = [f"Error: {e}"]

    return explanations


def create_feature_importance_plot(shap_values, feature_names, output_dir, config):
    if shap_values is None:
        return

    try:
        shap_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
        mean_abs_shap = np.abs(shap_array).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True).tail(config.explainability.top_n_features)

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        logger.info("Created feature importance plot")

    except Exception as e:
        logger.error(f"Feature importance plot failed: {e}")


def main():
    config = get_config()
    latest_run_id = get_latest_run_id(config.paths.artifacts_dir)

    if latest_run_id is None:
        logging.getLogger(__name__).error("No training runs found — run train.py first")
        return

    artifact_dir = config.paths.artifacts_dir / latest_run_id

    run_logger = setup_logging(config, latest_run_id)
    run_logger.info("\n" + "=" * 80)
    run_logger.info("Explainability Pipeline")
    run_logger.info("=" * 80)
    run_logger.info(f"Using artifacts from: {latest_run_id}")

    set_random_seeds(RANDOM_SEED)
    start_time = time.time()

    try:
        # load artifacts
        run_logger.info("\n--- Loading Artifacts ---")
        artifacts = load_artifacts(artifact_dir)

        if 'best_model' not in artifacts:
            run_logger.error("Best model not found in artifacts")
            return

        best_model = artifacts['best_model']
        feature_metadata = artifacts['feature_metadata']

        X_train = artifacts.get('X_train')
        X_test = artifacts.get('X_test')

        if X_train is None or X_test is None:
            run_logger.warning("X_train/X_test not found — re-run train.py")
            return

        if not SHAP_AVAILABLE:
            run_logger.error("SHAP not installed — pip install shap")
            with open(artifact_dir / 'explanations.txt', 'w') as f:
                f.write("SHAP not available\nInstall with: pip install shap\n")
            return

        # compute SHAP values (or load from cache if model + data unchanged)
        run_logger.info("\n--- Computing SHAP Values ---")
        shap_values, feature_names = load_shap_cache(artifact_dir, X_test)

        if shap_values is None:
            explainer = create_shap_explainer(best_model, X_train, config)
            shap_values = compute_shap_values(
                explainer, best_model, X_test, config,
                max_samples=config.explainability.shap_background_size
            )

            # get feature names from preprocessor
            try:
                preprocessor = best_model.named_steps.get('preprocessor')
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                n_features = shap_values.values.shape[1] if shap_values is not None else len(X_test.columns)
                feature_names = [f'feature_{i}' for i in range(n_features)]

            save_shap_cache(artifact_dir, X_test, shap_values, feature_names)

        # plots
        run_logger.info("\n--- Creating Plots ---")
        create_shap_plots(shap_values, feature_names, artifact_dir, config)
        create_feature_importance_plot(shap_values, feature_names, artifact_dir, config)

        # text explanations
        y_pred = best_model.predict(X_test)
        explanations = generate_text_explanations(
            shap_values, feature_names, X_test, y_pred, config
        )

        with open(artifact_dir / 'explanations.txt', 'w') as f:
            f.write("SHAP Prediction Explanations\n")
            f.write("=" * 80 + "\n\n")
            for exp in explanations:
                f.write(exp + "\n")

        with open(artifact_dir / 'explainability_summary.txt', 'w') as f:
            f.write("Explainability Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {artifacts.get('best_model_name', 'Unknown')}\n")
            f.write(f"Feature count: {len(feature_names)}\n")
            f.write(f"Test samples: {len(X_test)}\n\n")
            f.write("Outputs:\n")
            f.write("  - shap_beeswarm.png\n  - shap_bar.png\n")
            f.write("  - feature_importance.png / .csv\n")
            f.write(f"  - {config.explainability.n_local_examples} waterfall plots\n")
            f.write("  - explanations.txt\n")

        elapsed = time.time() - start_time
        run_logger.info("\n" + "=" * 80)
        run_logger.info("EXPLAINABILITY COMPLETE")
        run_logger.info(f"Total time: {format_duration(elapsed)}")

    except Exception as e:
        run_logger.error(f"Explainability pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
