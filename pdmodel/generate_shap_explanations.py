"""
SHAP Explanation Generator for PD Diagnosis Models
This script generates SHAP explanations for Parkinson's Disease diagnosis predictions.

Usage:
    python generate_shap_explanations.py --input /path/to/all_feature.npy --age 65 --gender 1 --output /path/to/output
    python generate_shap_explanations.py --manual --age 65 --gender 1 --output /path/to/output
"""

import os
import sys
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path to import pdModel modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import shap
except ImportError:
    print("ERROR: SHAP library not installed. Please install it with: pip install shap")
    sys.exit(1)

# Model configuration (adjust paths as needed)
MODEL_PATHS = "/home/pdapp/pd_api_server/api/pdModel/PD_pretrained_models/"
if not os.path.exists(MODEL_PATHS):
    # Try relative path from this file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATHS = os.path.join(current_dir, "PD_pretrained_models", "")
    if not os.path.exists(MODEL_PATHS):
        print(f"WARNING: Model path not found: {MODEL_PATHS}")
        print("Please update MODEL_PATHS in the script or ensure models are in the correct location.")

TRAINED_MODELS_LS = ["AdaBoost"]

# Feature names
gait_feature_name = [
    'left_foot_ground', 'right_foot_ground', 'left_right_foot_len_average', 'left_right_foot_len_max',
    'left_turning_duration', 'left_turning_slope', 'right_turning_duration', 'right_turning_slope',
    'l_leg_max_angles', 'l_leg_min_angles', 'r_leg_max_angles', 'r_leg_min_angles', 'l_arm_max_angles',
    'l_arm_min_angles', 'r_arm_max_angles', 'r_arm_min_angles', 'core_max_angles', "core_min_angles",
    'average_duration_per_rounds', "duration_change", "l_mean_steps", "r_mean_steps"
]

hand_features_name = [
    "Right Tapping Time", "Right Tapping Time Change", "Right Tapping Distance", "Left Tapping Time",
    "Left Tapping Time Change", "Left Tapping Distance", "Left Tapping Frequency", 
    "Left Tapping Intensity", "Left Tapping Power", "Left Tapping Frequency Change",
    "Right Tapping Frequency", "Right Tapping Intensity", "Right Tapping Power", "Right Tapping Frequency Change"
]

voice_feature_name = ['reading_time', 'score', 'pause(%)', 'volume change', 'pitch change', 'Average pitch']


def load_feature_selection_indices():
    """Load feature selection indices"""
    sfs_idx_path = os.path.join(MODEL_PATHS, 'nvp_sfs_idx.txt')
    if not os.path.exists(sfs_idx_path):
        raise FileNotFoundError(f"Feature selection indices not found: {sfs_idx_path}")
    return joblib.load(sfs_idx_path)


def load_models(modal, fold=10):
    """Load all fold models for a given modality"""
    save_file_dir = os.path.join(MODEL_PATHS, f'Save_model_{modal}', '')
    models = []
    
    for i in range(fold):
        clf_pth = os.path.join(save_file_dir, f'{i}_{TRAINED_MODELS_LS[0]}.joblib')
        if os.path.exists(clf_pth):
            models.append(joblib.load(clf_pth))
        else:
            print(f"WARNING: Model file not found: {clf_pth}")
    
    if not models:
        raise ValueError(f"No models found for {modal} modality")
    
    return models


def prepare_features(all_features, age, gender, modal):
    """Prepare features for a specific modality"""
    gait_len = len(gait_feature_name)
    hand_len = len(hand_features_name)
    voice_len = len(voice_feature_name)
    
    # Ensure all_features is a numpy array and flatten it
    all_features = np.array(all_features).flatten()
    
    if modal == "gait":
        feature_names = ['age', 'gender'] + gait_feature_name
        feature_data = np.concatenate([np.array([age, gender]), all_features[:gait_len]])
    elif modal == "hand":
        feature_names = ['age', 'gender'] + hand_features_name
        feature_data = np.concatenate([np.array([age, gender]), all_features[gait_len:gait_len + hand_len]])
    elif modal == "voice":
        feature_names = ['age', 'gender'] + voice_feature_name
        feature_data = np.concatenate([np.array([age, gender]), all_features[-voice_len:]])
    else:
        raise ValueError(f"Unknown modal: {modal}")
    
    # Ensure feature_data is 1D numpy array
    feature_data = np.array(feature_data).flatten()
    
    return feature_data, feature_names


# def explain_modality(all_features, age, gender, modal, sfs_idx, out_dir):
#     """Generate SHAP explanation for a single modality"""
#     print(f"\n{'='*60}")
#     print(f"Processing {modal.upper()} modality...")
#     print(f"{'='*60}")
    
#     # Prepare features
#     feature_data, feature_names = prepare_features(all_features, age, gender, modal)
    
#     # Get selected feature indices
#     f_idx = sfs_idx[f'{modal}_sfs_idx'][TRAINED_MODELS_LS[0]]
    
#     # Convert f_idx to numpy array and ensure it's the right format
#     if isinstance(f_idx, (list, tuple)):
#         f_idx = np.array(f_idx, dtype=int)
#     elif isinstance(f_idx, np.ndarray):
#         f_idx = f_idx.astype(int)
#     else:
#         # If it's a scalar, wrap it in an array
#         f_idx = np.array([int(f_idx)])
    
#     # Ensure feature_data is 1D numpy array
#     feature_data = np.array(feature_data).flatten()
    
#     # Validate indices are within bounds
#     if len(f_idx) > 0:
#         if np.any(f_idx < 0) or np.any(f_idx >= len(feature_data)):
#             raise ValueError(
#                 f"Invalid feature indices for {modal}. "
#                 f"Indices range: [{f_idx.min()}, {f_idx.max()}], "
#                 f"but feature_data length is {len(feature_data)}"
#             )
    
#     # Get selected feature names and values
#     selected_feature_names = [feature_names[int(i)] for i in f_idx]
#     selected_features = feature_data[f_idx]
    
#     print(f"Selected {len(selected_features)} features from {len(feature_data)} total features")
    
#     # Prepare data for SHAP (2D array)
#     X_explain = selected_features.reshape(1, -1)
    
#     # Load models
#     models = load_models(modal)
#     print(f"Loaded {len(models)} model(s) for {modal}")
    
#     # Create TreeExplainer using the first model
#     explainer = shap.TreeExplainer(models[0])
    
#     # Calculate SHAP values
#     print("Calculating SHAP values...")
#     shap_values = explainer.shap_values(X_explain)
    
#     # Handle binary classification output
#     if isinstance(shap_values, list):
#         shap_values = shap_values[1]  # Get PD class (class 1)
    
#     # Get base value
#     base_value = explainer.expected_value
#     if isinstance(base_value, np.ndarray):
#         base_value = base_value[1]
    
#     # Calculate prediction
#     prediction = base_value + shap_values.sum()
    
#     print(f"Base value (expected): {base_value:.4f}")
#     print(f"Prediction (PD probability): {prediction:.4f} ({prediction*100:.2f}%)")
    
#     # Create feature importance DataFrame
#     feature_importance_df = pd.DataFrame({
#         'feature': selected_feature_names,
#         'shap_value': shap_values[0],
#         'feature_value': X_explain[0],
#         'abs_shap': np.abs(shap_values[0])
#     })
#     feature_importance_df = feature_importance_df.sort_values('abs_shap', ascending=False)
    
#     # Save feature importance to CSV
#     csv_path = os.path.join(out_dir, f'shap_feature_importance_{modal}.csv')
#     feature_importance_df.to_csv(csv_path, index=False)
#     print(f"Saved feature importance to: {csv_path}")
    
#     # Generate visualizations
#     plot_shap_explanations(
#         shap_values[0],
#         X_explain[0],
#         selected_feature_names,
#         modal,
#         out_dir,
#         base_value,
#         prediction
#     )
    
#     # Create explanation summary
#     explanation = {
#         'modal': modal,
#         'prediction': float(prediction),
#         'base_value': float(base_value),
#         'feature_names': selected_feature_names,
#         'shap_values': shap_values[0].tolist(),
#         'feature_values': X_explain[0].tolist(),
#         'top_features_pd': feature_importance_df[feature_importance_df['shap_value'] > 0].head(5).to_dict('records'),
#         'top_features_normal': feature_importance_df[feature_importance_df['shap_value'] < 0].head(5).to_dict('records')
#     }
    
#     return explanation

def explain_modality(all_features, age, gender, modal, sfs_idx, out_dir):
    """Generate SHAP explanation for a single modality"""
    print(f"\n{'='*60}")
    print(f"Processing {modal.upper()} modality...")
    print(f"{'='*60}")
    
    # Prepare features
    feature_data, feature_names = prepare_features(all_features, age, gender, modal)
    
    # Get selected feature indices
    f_idx = sfs_idx[f'{modal}_sfs_idx'][TRAINED_MODELS_LS[0]]
    if isinstance(f_idx, (list, tuple)):
        f_idx = np.array(f_idx, dtype=int)
    elif isinstance(f_idx, np.ndarray):
        f_idx = f_idx.astype(int)
    else:
        f_idx = np.array([int(f_idx)])
    
    feature_data = np.array(feature_data).flatten()
    if len(f_idx) > 0:
        if np.any(f_idx < 0) or np.any(f_idx >= len(feature_data)):
            raise ValueError(
                f"Invalid feature indices for {modal}. "
                f"Indices range: [{f_idx.min()}, {f_idx.max()}], "
                f"feature_data length: {len(feature_data)}"
            )
    
    selected_feature_names = [feature_names[int(i)] for i in f_idx]
    selected_features = feature_data[f_idx]
    print(f"Selected {len(selected_features)} features from {len(feature_data)} total features")
    
    # Prepare data for SHAP
    X_explain = selected_features.reshape(1, -1)
    models = load_models(modal)
    print(f"Loaded {len(models)} model(s) for {modal}")
    
    model0 = models[0]
    model_class = type(model0).__name__.lower()
    # Choose explainer
    if "forest" in model_class or "gbm" in model_class or "tree" in model_class:
        explainer = shap.TreeExplainer(model0)
    else:
        # KernelExplainer for non-tree (e.g., SVC)
        # Background: use the explain sample itself or a small zero baseline
        background = np.zeros_like(X_explain)
        explainer = shap.KernelExplainer(model0.predict_proba, background)
    
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_explain)
    # For binary classification, shap_values could be list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Base value
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if isinstance(base_value, list) else base_value[1]
    
    prediction = base_value + shap_values.sum()
    print(f"Base value (expected): {base_value:.4f}")
    print(f"Prediction (PD probability): {prediction:.4f} ({prediction*100:.2f}%)")
    
    feature_importance_df = pd.DataFrame({
        'feature': selected_feature_names,
        'shap_value': shap_values[0],
        'feature_value': X_explain[0],
        'abs_shap': np.abs(shap_values[0])
    }).sort_values('abs_shap', ascending=False)
    
    csv_path = os.path.join(out_dir, f'shap_feature_importance_{modal}.csv')
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"Saved feature importance to: {csv_path}")
    
    plot_shap_explanations(
        shap_values[0],
        X_explain[0],
        selected_feature_names,
        modal,
        out_dir,
        base_value,
        prediction
    )
    
    explanation = {
        'modal': modal,
        'prediction': float(prediction),
        'base_value': float(base_value),
        'feature_names': selected_feature_names,
        'shap_values': shap_values[0].tolist(),
        'feature_values': X_explain[0].tolist(),
        'top_features_pd': feature_importance_df[feature_importance_df['shap_value'] > 0].head(5).to_dict('records'),
        'top_features_normal': feature_importance_df[feature_importance_df['shap_value'] < 0].head(5).to_dict('records')
    }
    return explanation
    


def plot_shap_explanations(shap_values, feature_values, feature_names, modal, out_dir, base_value, prediction):
    """Generate multiple SHAP visualization plots"""
    
    # 1. Waterfall plot
    print(f"Generating waterfall plot for {modal}...")
    plt.figure(figsize=(14, 10))
    try:
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=feature_values,
                feature_names=feature_names
            ),
            show=False,
            max_display=20
        )
        plt.title(f'{modal.capitalize()} Prediction Explanation\nPD Probability: {prediction:.2%}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shap_waterfall_{modal}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: shap_waterfall_{modal}.png")
    except Exception as e:
        print(f"  ✗ Error creating waterfall plot: {e}")
    
    # 2. Bar plot (horizontal)
    print(f"Generating bar plot for {modal}...")
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.35)))
    
    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(20, len(feature_names))  # Show top 20 features
    
    sorted_shap = shap_values[sorted_idx][:top_n]
    sorted_names = [feature_names[i] for i in sorted_idx[:top_n]]
    
    colors = ['#ff4444' if v > 0 else '#4444ff' for v in sorted_shap]
    bars = plt.barh(range(top_n), sorted_shap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    plt.yticks(range(top_n), sorted_names, fontsize=9)
    plt.xlabel('SHAP Value (Contribution to PD Probability)', fontsize=12, fontweight='bold')
    plt.title(f'{modal.capitalize()} Feature Contributions\n'
              f'Positive (Red) = pushes toward PD | Negative (Blue) = pushes toward Normal\n'
              f'Prediction: {prediction:.2%} PD probability',
              fontsize=13, fontweight='bold', pad=15)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    plt.grid(axis='x', alpha=0.3, linestyle=':')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_shap)):
        width = bar.get_width()
        label_x = width + (0.01 if width > 0 else -0.01)
        plt.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', fontsize=8, verticalalignment='center',
                horizontalalignment='left' if width > 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'shap_bar_{modal}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_bar_{modal}.png")
    
    # 3. Summary plot (scatter)
    print(f"Generating summary plot for {modal}...")
    plt.figure(figsize=(10, max(8, len(feature_names) * 0.4)))
    
    top_n = min(15, len(feature_names))
    sorted_idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    
    y_positions = range(top_n)
    shap_vals = shap_values[sorted_idx]
    feat_vals = feature_values[sorted_idx]
    feat_names = [feature_names[i] for i in sorted_idx]
    
    colors = ['#ff4444' if v > 0 else '#4444ff' for v in shap_vals]
    plt.scatter(shap_vals, y_positions, s=200, c=colors, alpha=0.7, 
               edgecolors='black', linewidth=1.5, zorder=3)
    
    # Add feature value as text
    for i, (shap_val, feat_val, name) in enumerate(zip(shap_vals, feat_vals, feat_names)):
        plt.text(shap_val + (0.015 if shap_val > 0 else -0.015), i, 
                f"{name}\n(value: {feat_val:.3f})", 
                fontsize=8, verticalalignment='center',
                horizontalalignment='left' if shap_val > 0 else 'right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.yticks(y_positions, [f"#{i+1}" for i in range(top_n)], fontsize=9)
    plt.xlabel('SHAP Value', fontsize=12, fontweight='bold')
    plt.title(f'{modal.capitalize()} Top {top_n} Feature Contributions\n'
              f'Prediction: {prediction:.2%} PD probability | Base: {base_value:.4f}',
              fontsize=13, fontweight='bold', pad=15)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    plt.grid(axis='x', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'shap_summary_{modal}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_summary_{modal}.png")


def explain_all_modalities(all_features, age, gender, sfs_idx, out_dir, modalities=None):
    """Generate SHAP explanations for all or specified modalities"""
    if modalities is None:
        modalities = ["gait", "hand", "voice"]
    
    explanations = {}
    
    for modal in modalities:
        try:
            explanations[modal] = explain_modality(all_features, age, gender, modal, sfs_idx, out_dir)
        except Exception as e:
            print(f"ERROR processing {modal}: {e}")
            explanations[modal] = {"error": str(e)}
            import traceback
            traceback.print_exc()
    
    # Create combined visualization
    if len([e for e in explanations.values() if 'error' not in e]) > 1:
        create_combined_plot(explanations, out_dir)
    
    return explanations


def create_combined_plot(explanations, out_dir):
    """Create a combined visualization showing all modalities"""
    print("\nGenerating combined visualization...")
    
    valid_modals = [m for m, e in explanations.items() if 'error' not in e]
    if len(valid_modals) == 0:
        return
    
    fig, axes = plt.subplots(1, len(valid_modals), figsize=(6*len(valid_modals), 8))
    if len(valid_modals) == 1:
        axes = [axes]
    
    for idx, modal in enumerate(valid_modals):
        exp = explanations[modal]
        ax = axes[idx]
        
        feature_names = exp['feature_names']
        shap_values = np.array(exp['shap_values'])
        
        # Get top 10 features
        top_indices = np.argsort(np.abs(shap_values))[-10:][::-1]
        
        colors = ['#ff4444' if shap_values[i] > 0 else '#4444ff' for i in top_indices]
        bars = ax.barh(range(len(top_indices)), shap_values[top_indices], color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
        ax.set_xlabel('SHAP Value', fontsize=11, fontweight='bold')
        ax.set_title(f'{modal.capitalize()}\nPD Prob: {exp["prediction"]:.2%}', 
                     fontsize=12, fontweight='bold', pad=10)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        
        # Add value labels
        for bar, val in zip(bars, shap_values[top_indices]):
            width = bar.get_width()
            ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', fontsize=7, verticalalignment='center',
                   horizontalalignment='left' if width > 0 else 'right')
    
    plt.suptitle('PD Diagnosis Feature Explanations Across All Modalities', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_combined.png")


def load_features_from_file(file_path):
    """Load features from .npy file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    
    features = np.load(file_path)
    print(f"Loaded features from: {file_path}")
    print(f"Feature shape: {features.shape}")
    
    # Expected total features: gait (22) + hand (14) + voice (6) = 42
    expected_len = len(gait_feature_name) + len(hand_features_name) + len(voice_feature_name)
    if len(features) != expected_len:
        print(f"WARNING: Expected {expected_len} features, got {len(features)}")
    
    return features


def input_features_manually():
    """Manually input features"""
    print("\n" + "="*60)
    print("MANUAL FEATURE INPUT")
    print("="*60)
    print("\nPlease enter features in order:")
    print(f"Gait features ({len(gait_feature_name)}):")
    for i, name in enumerate(gait_feature_name, 1):
        print(f"  {i}. {name}")
    print(f"\nHand features ({len(hand_features_name)}):")
    for i, name in enumerate(hand_features_name, 1):
        print(f"  {i}. {name}")
    print(f"\nVoice features ({len(voice_feature_name)}):")
    for i, name in enumerate(voice_feature_name, 1):
        print(f"  {i}. {name}")
    
    all_features = []
    
    print("\n--- Gait Features ---")
    for name in gait_feature_name:
        val = input(f"{name}: ").strip()
        try:
            all_features.append(float(val))
        except ValueError:
            print(f"Invalid input, using 0.0")
            all_features.append(0.0)
    
    print("\n--- Hand Features ---")
    for name in hand_features_name:
        val = input(f"{name}: ").strip()
        try:
            all_features.append(float(val))
        except ValueError:
            print(f"Invalid input, using 0.0")
            all_features.append(0.0)
    
    print("\n--- Voice Features ---")
    for name in voice_feature_name:
        val = input(f"{name}: ").strip()
        try:
            all_features.append(float(val))
        except ValueError:
            print(f"Invalid input, using 0.0")
            all_features.append(0.0)
    
    return np.array(all_features)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SHAP explanations for PD diagnosis models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using saved feature file:
  python generate_shap_explanations.py --input /path/to/all_feature.npy --age 65 --gender 1 --output ./shap_results
  
  # Manual input:
  python generate_shap_explanations.py --manual --age 65 --gender 1 --output ./shap_results
  
  # Specific modality only:
  python generate_shap_explanations.py --input features.npy --age 65 --gender 1 --modal gait --output ./results
        """
    )
    
    parser.add_argument('--input', '-i', type=str, help='Path to all_feature.npy file')
    parser.add_argument('--manual', '-m', action='store_true', help='Manually input features')
    parser.add_argument('--age', '-a', type=int, required=True, help='Patient age')
    parser.add_argument('--gender', '-g', type=int, required=True, 
                       help='Patient gender (0=non, 1=male, 2=female)')
    parser.add_argument('--output', '-o', type=str, default='./shap_output', 
                       help='Output directory for results (default: ./shap_output)')
    parser.add_argument('--modal', type=str, choices=['gait', 'hand', 'voice', 'all'], 
                       default='all', help='Modality to explain (default: all)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Custom path to model directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.manual and not args.input:
        parser.error("Either --input or --manual must be specified")
    
    if args.manual and args.input:
        parser.error("Cannot use both --input and --manual")
    
    # Set model path if provided
    global MODEL_PATHS
    if args.model_path:
        MODEL_PATHS = args.model_path
        if not MODEL_PATHS.endswith('/'):
            MODEL_PATHS += '/'
    
    # Create output directory
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")
    
    # Load features
    if args.manual:
        all_features = input_features_manually()
    else:
        all_features = load_features_from_file(args.input)
    
    # Load feature selection indices
    print("\nLoading feature selection indices...")
    try:
        sfs_idx = load_feature_selection_indices()
        print("✓ Feature indices loaded")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Determine modalities to process
    if args.modal == 'all':
        modalities = ["gait", "hand", "voice"]
    else:
        modalities = [args.modal]
    
    # Generate explanations
    print(f"\n{'='*60}")
    print("GENERATING SHAP EXPLANATIONS")
    print(f"{'='*60}")
    print(f"Age: {args.age}")
    print(f"Gender: {args.gender}")
    print(f"Modalities: {', '.join(modalities)}")
    
    try:
        explanations = explain_all_modalities(
            all_features, args.age, args.gender, sfs_idx, out_dir, modalities
        )
        
        # Save explanations to JSON
        json_path = os.path.join(out_dir, 'shap_explanations.json')
        with open(json_path, 'w') as f:
            json.dump(explanations, f, indent=2, default=str)
        print(f"\n✓ Saved explanations to: {json_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for modal, exp in explanations.items():
            if 'error' not in exp:
                print(f"\n{modal.upper()}:")
                print(f"  PD Probability: {exp['prediction']:.2%}")
                print(f"  Base Value: {exp['base_value']:.4f}")
                print(f"\n  Top 5 Features Pushing Toward PD:")
                for feat in exp['top_features_pd']:
                    print(f"    - {feat['feature']}: {feat['shap_value']:.4f}")
                print(f"\n  Top 5 Features Pushing Toward Normal:")
                for feat in exp['top_features_normal']:
                    print(f"    - {feat['feature']}: {feat['shap_value']:.4f}")
            else:
                print(f"\n{modal.upper()}: ERROR - {exp['error']}")
        
        print(f"\n{'='*60}")
        print(f"All results saved to: {out_dir}")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())