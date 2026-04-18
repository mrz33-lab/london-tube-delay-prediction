"""
Classification evaluation pipeline — binary and ordinal classifiers.

Complements the regression pipeline in train.py by training classifiers on
the same feature set and evaluating with classification-specific metrics:
F1, Precision, Recall, ROC-AUC, calibration curves, and time-window analysis.

Run standalone:
    python classify.py

Or called automatically at the end of train.main() via run_classification_evaluation().
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize

from config import get_config, Config, RANDOM_SEED
from features import create_preprocessing_pipeline

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def create_binary_labels(delays: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Binary: 1 = delayed (severity >= threshold), 0 = not delayed.

    The target column is delay_severity (0=Good, 1=Minor, 2=Severe), so the
    default threshold of 1.0 maps any delay (minor or severe) to class 1.
    """
    return (delays >= threshold).astype(int)


def create_ordinal_labels(
    delays: np.ndarray,
    good_max: float = 0.5,
    minor_max: float = 1.5,
) -> np.ndarray:
    """Ordinal: 'Good Service' / 'Minor Delays' / 'Severe Delays'.

    Thresholds are midpoints between severity integer values (0, 1, 2) so
    that 0 → Good Service, 1 → Minor Delays, 2 → Severe Delays.
    """
    labels = np.empty(len(delays), dtype=object)
    labels[delays <= good_max] = 'Good Service'
    labels[(delays > good_max) & (delays <= minor_max)] = 'Minor Delays'
    labels[delays > minor_max] = 'Severe Delays'
    return labels


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_logistic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    config: Config,
    task: str = 'binary',
) -> Pipeline:
    """Logistic Regression (direct fit on full training set)."""
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=RANDOM_SEED,
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(X_train, y_train)
    return pipeline


def _train_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    config: Config,
    task: str = 'binary',
) -> Pipeline:
    """LightGBM classifier."""
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    objective = 'binary' if task == 'binary' else 'multiclass'
    clf = lgb.LGBMClassifier(
        objective=objective,
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_SEED,
        n_estimators=100,
        num_leaves=31,
        min_child_samples=5,
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    task: str,
    class_names: List[str],
) -> Dict:
    """Compute classification metrics for a trained model."""
    y_pred = model.predict(X_test)
    metrics: Dict = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    }

    # ROC-AUC (binary or OvR)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if task == 'binary':
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob[:, 1]))
            else:
                y_test_bin = label_binarize(y_test, classes=class_names)
                metrics['roc_auc_ovr'] = float(roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr'))
    except Exception as e:
        logger.warning('Could not compute ROC-AUC: %s', e)

    # Per-class metrics — supply explicit labels so the report works even when
    # some classes are absent from the test split (e.g. small synthetic sets).
    report = classification_report(
        y_test, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics['per_class'] = {k: v for k, v in report.items() if k in class_names}

    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    # Let sklearn determine the label order from y_true/y_pred so that integer
    # binary labels (0/1) and string ordinal labels both work without mismatch.
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _calibration_analysis(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    task: str,
    class_names: List[str],
    output_dir: Path,
) -> Dict:
    """Platt scaling + isotonic calibration with reliability diagrams."""
    results: Dict = {}

    if task != 'binary':
        # Calibration plots only supported for binary task here
        return results

    for method in ('sigmoid', 'isotonic'):
        try:
            cal_model = CalibratedClassifierCV(model, method=method, cv=3)
            cal_model.fit(X_train, y_train)
            cal_metrics = _evaluate_classifier(cal_model, X_test, y_test, task, class_names)
            results[method] = cal_metrics
            logger.info('  Calibrated (%s) F1: %.3f', method, cal_metrics.get('f1_weighted', float('nan')))
        except Exception as e:
            logger.warning('Calibration (%s) failed: %s', method, e)
            results[method] = {'error': str(e)}

    # Reliability diagram (binary, uncalibrated)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(prob_pred, prob_true, 's-', label='Uncalibrated', color='#003688')
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title('Reliability Diagram (Binary)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / 'calibration_reliability.png', dpi=150)
            plt.close(fig)
    except Exception as e:
        logger.warning('Reliability diagram failed: %s', e)

    return results


def _time_window_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    timestamps: pd.Series,
    task: str,
    class_names: List[str],
    n_windows: int = 16,
) -> List[Dict]:
    """Split test set into temporal windows and report metrics per window."""
    window_size = max(1, len(X_test) // n_windows)
    records = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, len(X_test))
        if end_idx <= start_idx:
            break
        X_w = X_test.iloc[start_idx:end_idx]
        y_w = y_test[start_idx:end_idx]
        ts_w = timestamps.iloc[start_idx:end_idx] if timestamps is not None else None

        if len(np.unique(y_w)) < 2:
            continue

        y_pred_w = model.predict(X_w)
        record = {
            'window': i,
            'start': str(ts_w.iloc[0]) if ts_w is not None else start_idx,
            'end': str(ts_w.iloc[-1]) if ts_w is not None else end_idx,
            'n': len(y_w),
            'f1_weighted': float(f1_score(y_w, y_pred_w, average='weighted', zero_division=0)),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_classification_evaluation(
    artifact_dir: Path,
    config: Config,
    X_train: pd.DataFrame,
    y_train_reg: np.ndarray,
    X_test: pd.DataFrame,
    y_test_reg: np.ndarray,
    test_timestamps: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    test_preds: Optional[np.ndarray] = None,
    y_train_status: Optional[np.ndarray] = None,
    y_test_status: Optional[np.ndarray] = None,
) -> None:
    """Run the full classification evaluation pipeline.

    Can be called from train.py with data already in memory,
    or standalone by loading from artifacts.

    y_train_status / y_test_status: raw TfL status labels ('Good Service',
        'Minor Delays', 'Severe Delays') aligned to X_train / X_test.  When
        provided, the ordinal task uses these directly instead of re-deriving
        labels from delay_minutes.  This eliminates the circularity present
        in the fallback path: for real data, delay_minutes is a proxy derived
        FROM the status label, so mapping delay → status just inverts a known
        function rather than learning a genuine predictor.
    """
    print('=' * 80)
    print('CLASSIFICATION EVALUATION PIPELINE')

    output_dir = artifact_dir / 'classification'
    output_dir.mkdir(exist_ok=True)

    all_results: Dict = {}

    # --- Binary task ---
    print('\n--- Binary Classification (Delayed / Not Delayed) ---')
    y_train_bin = create_binary_labels(y_train_reg)
    y_test_bin = create_binary_labels(y_test_reg)
    logger.info('Binary train distribution: %s', dict(zip(*np.unique(y_train_bin, return_counts=True))))
    logger.info('Binary test distribution:  %s', dict(zip(*np.unique(y_test_bin, return_counts=True))))

    bin_class_names = ['Not Delayed', 'Delayed']

    lr_bin = _train_logistic(X_train, y_train_bin, numeric_features, categorical_features, config, 'binary')
    lr_bin_metrics = _evaluate_classifier(lr_bin, X_test, y_test_bin, 'binary', bin_class_names)
    logger.info('Logistic Regression binary F1: %.3f', lr_bin_metrics.get('f1_weighted', float('nan')))

    y_pred_bin = lr_bin.predict(X_test)
    _plot_confusion_matrix(y_test_bin, y_pred_bin, bin_class_names,
                           'Binary Confusion Matrix (LR)', output_dir / 'cm_binary_lr.png')
    cal_results = _calibration_analysis(lr_bin, X_train, y_train_bin, X_test, y_test_bin,
                                        'binary', bin_class_names, output_dir)
    tw_results = _time_window_analysis(lr_bin, X_test, y_test_bin, test_timestamps,
                                       'binary', bin_class_names)

    all_results['binary_logistic'] = {
        'metrics': lr_bin_metrics,
        'calibration': cal_results,
        'time_windows': tw_results,
    }

    if LIGHTGBM_AVAILABLE:
        lgbm_bin = _train_lgbm_classifier(X_train, y_train_bin, numeric_features,
                                           categorical_features, config, 'binary')
        lgbm_bin_metrics = _evaluate_classifier(lgbm_bin, X_test, y_test_bin, 'binary', bin_class_names)
        logger.info('LightGBM binary F1: %.3f', lgbm_bin_metrics.get('f1_weighted', float('nan')))
        y_pred_lgbm_bin = lgbm_bin.predict(X_test)
        _plot_confusion_matrix(y_test_bin, y_pred_lgbm_bin, bin_class_names,
                               'Binary Confusion Matrix (LGBM)', output_dir / 'cm_binary_lgbm.png')
        all_results['binary_lgbm'] = {'metrics': lgbm_bin_metrics}

    # --- Ordinal task ---
    print('\n--- Ordinal Classification (Good / Minor / Severe) ---')
    if y_train_status is not None and y_test_status is not None:
        # Preferred path: use actual TfL status labels saved by train.py.
        # For real data, status is the true TfL label and delay_minutes is a
        # proxy derived FROM it, so re-mapping delay → status would just invert
        # a known function rather than evaluate a genuine predictor.
        y_train_ord = y_train_status
        y_test_ord = y_test_status
        logger.info('Ordinal task: using raw TfL status labels (non-circular evaluation)')
    else:
        # Fallback: re-derive labels from delay proxy. This path is inherently
        # circular for real data and should only be used when status labels are
        # unavailable (e.g. evaluating a model trained before this fix).
        logger.warning(
            'Ordinal task: raw status labels not available — re-deriving from '
            'delay proxy. Results are subject to circularity. Re-run train.py '
            'to generate y_train_status.csv and y_test_status.csv.'
        )
        y_train_ord = create_ordinal_labels(
            y_train_reg,
            good_max=config.data.status_good_max,
            minor_max=config.data.status_minor_max,
        )
        y_test_ord = create_ordinal_labels(
            y_test_reg,
            good_max=config.data.status_good_max,
            minor_max=config.data.status_minor_max,
        )
    ord_class_names = ['Good Service', 'Minor Delays', 'Severe Delays']

    lr_ord = _train_logistic(X_train, y_train_ord, numeric_features, categorical_features, config, 'ordinal')
    lr_ord_metrics = _evaluate_classifier(lr_ord, X_test, y_test_ord, 'ordinal', ord_class_names)
    logger.info('Logistic Regression ordinal F1: %.3f', lr_ord_metrics.get('f1_weighted', float('nan')))

    y_pred_ord = lr_ord.predict(X_test)
    _plot_confusion_matrix(y_test_ord, y_pred_ord, ord_class_names,
                           'Ordinal Confusion Matrix (LR)', output_dir / 'cm_ordinal_lr.png')
    all_results['ordinal_logistic'] = {'metrics': lr_ord_metrics}

    if LIGHTGBM_AVAILABLE:
        lgbm_ord = _train_lgbm_classifier(X_train, y_train_ord, numeric_features,
                                           categorical_features, config, 'ordinal')
        lgbm_ord_metrics = _evaluate_classifier(lgbm_ord, X_test, y_test_ord, 'ordinal', ord_class_names)
        logger.info('LightGBM ordinal F1: %.3f', lgbm_ord_metrics.get('f1_weighted', float('nan')))
        all_results['ordinal_lgbm'] = {'metrics': lgbm_ord_metrics}

    # Save results
    results_path = output_dir / 'classification_results.json'
    with open(results_path, 'w') as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info('Classification results saved to %s', results_path)
    print(f'\nClassification results saved to {results_path}')


def main() -> None:
    """Run classification evaluation on the latest training run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    config = get_config()
    from utils import get_latest_run_id
    run_id = get_latest_run_id(Path('artifacts'))
    if run_id is None:
        print('No training runs found. Run `python train.py` first.')
        return

    artifact_dir = Path('artifacts') / run_id
    print(f'Using artifacts from: {artifact_dir}')

    # Load preprocessed data saved by train.py
    X_train = pd.read_parquet(artifact_dir / 'X_train.parquet')
    X_test = pd.read_parquet(artifact_dir / 'X_test.parquet')
    test_preds_df = pd.read_csv(artifact_dir / 'test_predictions.csv')

    # y_train is saved separately by train.py because prepare_features_for_model
    # excludes the target column from the X_train feature matrix.
    y_train_csv = artifact_dir / 'y_train.csv'
    if not y_train_csv.exists():
        print(
            f"ERROR: y_train.csv not found in {artifact_dir}\n"
            "Re-run train.py to regenerate artifacts."
        )
        return
    y_train_reg = pd.read_csv(y_train_csv)['delay_minutes'].values

    y_test_reg = test_preds_df['actual'].values
    test_timestamps = (
        test_preds_df['timestamp']
        if 'timestamp' in test_preds_df.columns
        else pd.Series(range(len(test_preds_df)))
    )

    # Load raw TfL status labels when available (preferred: non-circular ordinal task)
    y_train_status: Optional[np.ndarray] = None
    y_test_status: Optional[np.ndarray] = None
    status_train_csv = artifact_dir / 'y_train_status.csv'
    status_test_csv  = artifact_dir / 'y_test_status.csv'
    if status_train_csv.exists() and status_test_csv.exists():
        y_train_status = pd.read_csv(status_train_csv)['status'].values
        y_test_status  = pd.read_csv(status_test_csv)['status'].values
        print(f'Loaded raw status labels: {len(y_train_status)} train, {len(y_test_status)} test')
    else:
        print(
            'WARNING: y_train_status.csv / y_test_status.csv not found. '
            'Re-run train.py to generate them and eliminate ordinal-task circularity.'
        )

    # Use pd.api.types for robust dtype detection (handles extension dtypes)
    numeric_features = [
        c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical_features = [
        c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])
    ]

    run_classification_evaluation(
        artifact_dir=artifact_dir,
        config=config,
        X_train=X_train,
        y_train_reg=y_train_reg,
        X_test=X_test,
        y_test_reg=y_test_reg,
        test_timestamps=test_timestamps,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        y_train_status=y_train_status,
        y_test_status=y_test_status,
    )


if __name__ == '__main__':
    main()
