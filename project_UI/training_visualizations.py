import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
from sklearn.decomposition import PCA
from sklearn.tree import export_text
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.lines as mlines


################################ helper funcs ##########################################

def reduce_to_2d(X, feature_names=None):
    """Reduces high-dimensional data to 2D using PCA for visualization."""
    if X.shape[1] == 2:
        labels = feature_names[:2] if feature_names else ['Feature 1', 'Feature 2']
        return X, labels[0], labels[1]

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    return X_2d, f"PC1 ({explained_var[0]:.1%})", f"PC2 ({explained_var[1]:.1%})"


def get_best_feature_index(X, y):
    """Find the feature most correlated with target for 1D visualization."""
    correlations = []
    for i in range(X.shape[1]):
        corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        correlations.append(corr if not np.isnan(corr) else 0)
    return np.argmax(correlations)


# ============================================================================
# DECISION TREE VISUALIZATION
# ============================================================================

def visualize_decision_tree(model, X_train, y_train, feature_names, max_depth):
    """
    Animated decision tree building visualization.
    Shows tree nodes appearing one by one with split information.
    """
    st.markdown("#### Decision Tree Training Animation")
    st.caption("Watch how the tree splits the data to make predictions")

    # Get tree structure
    tree = model.tree_
    n_nodes = tree.node_count
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    n_samples = tree.n_node_samples
    values = tree.value

    # Build node info for animation
    nodes_info = []

    def build_node_info(node_id, depth, x_pos, parent_pos=None, is_left=None):
        if node_id == -1:
            return

        is_leaf = children_left[node_id] == children_right[node_id]

        if is_leaf:
            class_counts = values[node_id][0]
            predicted_class = np.argmax(class_counts)
            label = f"Predict:\n{'Died' if predicted_class == 0 else 'Survived'}\n(n={n_samples[node_id]})"
        else:
            feat_name = feature_names[feature[node_id]] if feature[node_id] < len(feature_names) else f"X[{feature[node_id]}]"
            label = f"{feat_name}\n<= {threshold[node_id]:.2f}\n(n={n_samples[node_id]})"

        nodes_info.append({
            'id': node_id,
            'depth': depth,
            'x': x_pos,
            'y': -depth,
            'label': label,
            'is_leaf': is_leaf,
            'parent_pos': parent_pos,
            'is_left': is_left,
            'samples': n_samples[node_id]
        })

        if not is_leaf:
            width = 2 ** (max_depth - depth - 1)
            build_node_info(children_left[node_id], depth + 1, x_pos - width/2, (x_pos, -depth), True)
            build_node_info(children_right[node_id], depth + 1, x_pos + width/2, (x_pos, -depth), False)

    # Build from root
    build_node_info(0, 0, 0)

    # Sort by depth for animation order
    nodes_info.sort(key=lambda x: (x['depth'], x['x']))

    # Animation
    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, node in enumerate(nodes_info):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-2**max_depth, 2**max_depth)
        ax.set_ylim(-max_depth - 1, 1)
        ax.axis('off')
        ax.set_title("Decision Tree Building Process", fontsize=14, fontweight='bold')

        # Draw all nodes up to current
        for j in range(i + 1):
            n = nodes_info[j]

            # Draw edge from parent
            if n['parent_pos'] is not None:
                ax.plot([n['parent_pos'][0], n['x']],
                       [n['parent_pos'][1], n['y']],
                       'k-', linewidth=1.5, alpha=0.6)
                # Add Yes/No labels
                mid_x = (n['parent_pos'][0] + n['x']) / 2
                mid_y = (n['parent_pos'][1] + n['y']) / 2
                label_text = "Yes" if n['is_left'] else "No"
                ax.text(mid_x, mid_y, label_text, fontsize=8, ha='center',
                       color='green' if n['is_left'] else 'red')

            # Draw node
            color = '#90EE90' if n['is_leaf'] else '#87CEEB'
            if j == i:  # Highlight current node
                color = '#FFD700'

            bbox = FancyBboxPatch((n['x'] - 1.5, n['y'] - 0.4), 3, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(bbox)
            ax.text(n['x'], n['y'], n['label'], ha='center', va='center',
                   fontsize=7, fontweight='bold' if j == i else 'normal')

        # Update status
        if node['is_leaf']:
            status_text.markdown(f"**Step {i+1}/{len(nodes_info)}**: Created leaf node - {node['samples']} samples")
        else:
            status_text.markdown(f"**Step {i+1}/{len(nodes_info)}**: Split on feature with {node['samples']} samples")

        placeholder.pyplot(fig)
        progress_bar.progress((i + 1) / len(nodes_info))
        plt.close(fig)
        time.sleep(0.5)

    status_text.success("Tree construction complete!")

    # Add legend
    st.markdown("""
    **Legend:**
    - **Blue boxes**: Decision nodes (split points)
    - **Green boxes**: Leaf nodes (predictions)
    - **Yes/No**: Direction based on condition
    """)


# ============================================================================
# LINEAR REGRESSION VISUALIZATION
# ============================================================================

def visualize_linear_regression(model, X_train, y_train, feature_names):
    """
    Animated linear regression showing line fitting process.
    Shows the best-fit line adjusting with residuals.
    """
    st.markdown("#### Linear Regression Training Animation")
    st.caption("Watch how the line adjusts to minimize error")

    # Convert to numpy arrays if needed (fixes pandas indexing issues)
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Get best feature for visualization
    best_feat_idx = get_best_feature_index(X_train_np, y_train_np)
    X_1d = X_train_np[:, best_feat_idx]
    feat_name = feature_names[best_feat_idx] if best_feat_idx < len(feature_names) else f"Feature {best_feat_idx}"

    # Get final coefficients
    final_slope = model.coef_[best_feat_idx]
    final_intercept = model.intercept_

    # Simulate gradient descent steps
    n_steps = 15

    # Start with random line and move toward optimal
    np.random.seed(42)
    start_slope = final_slope + np.random.uniform(-2, 2)
    start_intercept = final_intercept + np.random.uniform(-1, 1)

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()

    x_line = np.linspace(X_1d.min() - 0.5, X_1d.max() + 0.5, 100)

    for step in range(n_steps + 1):
        # Interpolate between start and final
        t = step / n_steps
        current_slope = start_slope + t * (final_slope - start_slope)
        current_intercept = start_intercept + t * (final_intercept - start_intercept)

        y_line = current_slope * x_line + current_intercept
        y_pred = current_slope * X_1d + current_intercept

        # Calculate MSE
        mse = np.mean((y_train_np - y_pred) ** 2)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        colors = ['#FF6B6B' if y == 0 else '#4ECDC4' for y in y_train_np]
        ax.scatter(X_1d, y_train_np, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Plot regression line
        line_color = '#FFD700' if step < n_steps else '#32CD32'
        ax.plot(x_line, y_line, color=line_color, linewidth=3,
               label=f'y = {current_slope:.3f}x + {current_intercept:.3f}')

        # Draw residual lines (for a subset of points)
        if step > 0:
            sample_indices = np.random.choice(len(X_1d), min(20, len(X_1d)), replace=False)
            for idx in sample_indices:
                ax.plot([X_1d[idx], X_1d[idx]], [y_train_np[idx], y_pred[idx]],
                       'r--', alpha=0.3, linewidth=1)

        ax.set_xlabel(feat_name, fontsize=12)
        ax.set_ylabel('Survived (0/1)', fontsize=12)
        ax.set_title(f'Linear Regression Fitting - Iteration {step}/{n_steps}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.set_ylim(-0.2, 1.2)

        placeholder.pyplot(fig)
        progress_bar.progress((step + 1) / (n_steps + 1))
        status_text.markdown(f"**Iteration {step}**: Adjusting line parameters...")
        metrics_placeholder.metric("Mean Squared Error", f"{mse:.4f}")

        plt.close(fig)
        time.sleep(0.4)

    status_text.success("Line fitting complete! Optimal parameters found.")

    st.markdown("""
    **What you saw:**
    - The line started at a random position
    - It gradually adjusted slope and intercept
    - Red dashed lines show residuals (errors)
    - MSE decreased as the line found the best fit
    """)


# ============================================================================
# KNN VISUALIZATION
# ============================================================================

def visualize_knn(model, X_train, y_train, feature_names, k):
    """
    Animated KNN visualization showing neighbor selection.
    Shows a test point and its K nearest neighbors being selected.
    """
    st.markdown("#### K-Nearest Neighbors Prediction Animation")
    st.caption("Watch how KNN finds neighbors to make a prediction")

    # Convert to numpy arrays if needed (fixes pandas indexing issues)
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Reduce to 2D for visualization
    X_2d, xlabel, ylabel = reduce_to_2d(X_train_np, feature_names)

    # Create a test point (centroid of the data with some offset)
    test_point = np.mean(X_2d, axis=0) + np.array([0.5, 0.3])

    # Calculate distances to all training points
    distances = np.sqrt(np.sum((X_2d - test_point) ** 2, axis=1))
    sorted_indices = np.argsort(distances)

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Animation phases
    phases = ['show_data', 'show_test', 'expand_circle'] + [f'neighbor_{i}' for i in range(k)] + ['vote', 'result']

    for phase_idx, phase in enumerate(phases):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all training points
        for i, (x, y, label) in enumerate(zip(X_2d[:, 0], X_2d[:, 1], y_train_np)):
            color = '#FF6B6B' if label == 0 else '#4ECDC4'
            alpha = 0.6
            size = 60

            # Highlight if this point is a selected neighbor
            if phase.startswith('neighbor_') or phase in ['vote', 'result']:
                neighbor_num = int(phase.split('_')[1]) if phase.startswith('neighbor_') else k
                if phase in ['vote', 'result']:
                    neighbor_num = k
                if i in sorted_indices[:neighbor_num]:
                    alpha = 1.0
                    size = 120
                    ax.plot([test_point[0], x], [test_point[1], y], 'k--', alpha=0.5, linewidth=1)

            ax.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)

        # Show test point
        if phase != 'show_data':
            ax.scatter(test_point[0], test_point[1], c='gold', s=200, marker='*',
                      edgecolors='black', linewidth=2, zorder=10, label='Test Point')

        # Draw expanding circle
        if phase == 'expand_circle':
            max_dist = distances[sorted_indices[k-1]]
            for r in np.linspace(0, max_dist, 5):
                circle = plt.Circle(test_point, r, fill=False, color='gray',
                                   linestyle='--', alpha=0.3)
                ax.add_patch(circle)

        # Draw final circle containing k neighbors
        if phase in ['vote', 'result'] or phase.startswith('neighbor_'):
            if phase.startswith('neighbor_'):
                n_neighbors = int(phase.split('_')[1]) + 1
            else:
                n_neighbors = k
            radius = distances[sorted_indices[n_neighbors-1]] * 1.1
            circle = plt.Circle(test_point, radius, fill=False, color='purple',
                               linestyle='-', linewidth=2, alpha=0.7)
            ax.add_patch(circle)

        # Show voting results
        if phase in ['vote', 'result']:
            neighbor_labels = y_train_np[sorted_indices[:k]]
            died_count = np.sum(neighbor_labels == 0)
            survived_count = np.sum(neighbor_labels == 1)

            # Add vote count box
            textstr = f'Votes:\nDied: {died_count}\nSurvived: {survived_count}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)

            if phase == 'result':
                prediction = 'Survived' if survived_count > died_count else 'Died'
                ax.set_title(f'KNN Prediction: {prediction}!', fontsize=14, fontweight='bold', color='green')
        else:
            ax.set_title(f'K-Nearest Neighbors (K={k})', fontsize=14, fontweight='bold')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Legend
        died_patch = mpatches.Patch(color='#FF6B6B', label='Died')
        survived_patch = mpatches.Patch(color='#4ECDC4', label='Survived')
        ax.legend(handles=[died_patch, survived_patch], loc='upper right')

        placeholder.pyplot(fig)
        progress_bar.progress((phase_idx + 1) / len(phases))

        # Status messages
        if phase == 'show_data':
            status_text.markdown("**Step 1**: Showing all training data points")
        elif phase == 'show_test':
            status_text.markdown("**Step 2**: New test point appears (gold star)")
        elif phase == 'expand_circle':
            status_text.markdown("**Step 3**: Finding distances to all points...")
        elif phase.startswith('neighbor_'):
            n = int(phase.split('_')[1]) + 1
            status_text.markdown(f"**Step {3+n}**: Found neighbor {n} of {k}")
        elif phase == 'vote':
            status_text.markdown(f"**Step {4+k}**: Neighbors vote on the prediction")
        elif phase == 'result':
            status_text.success("Prediction complete!")

        plt.close(fig)
        time.sleep(0.6)

    st.markdown("""
    **What you saw:**
    - KNN doesn't "train" - it memorizes all data points
    - For a new point, it finds the K closest neighbors
    - Neighbors vote: majority class wins
    - Smaller K = more sensitive to noise, Larger K = smoother boundaries
    """)


# ============================================================================
# LOGISTIC REGRESSION VISUALIZATION
# ============================================================================

def visualize_logistic_regression(model, X_train, y_train, feature_names, C):
    """
    Animated logistic regression showing decision boundary forming.
    """
    st.markdown("#### Logistic Regression Training Animation")
    st.caption("Watch the decision boundary separate the classes")

    # Convert to numpy arrays if needed (fixes pandas indexing issues)
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Reduce to 2D
    X_2d, xlabel, ylabel = reduce_to_2d(X_train_np, feature_names)

    # Get model coefficients (retrain on 2D data for visualization)
    from sklearn.linear_model import LogisticRegression
    lr_2d = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr_2d.fit(X_2d, y_train_np)

    final_coef = lr_2d.coef_[0]
    final_intercept = lr_2d.intercept_[0]

    # Simulate optimization
    n_steps = 12
    np.random.seed(42)
    start_coef = final_coef + np.random.uniform(-2, 2, size=2)
    start_intercept = final_intercept + np.random.uniform(-1, 1)

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    for step in range(n_steps + 1):
        t = step / n_steps
        current_coef = start_coef + t * (final_coef - start_coef)
        current_intercept = start_intercept + t * (final_intercept - start_intercept)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot data points
        colors = ['#FF6B6B' if y == 0 else '#4ECDC4' for y in y_train_np]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=60, alpha=0.7,
                  edgecolors='black', linewidth=0.5)

        # Plot decision boundary
        xx = np.linspace(x_min, x_max, 100)
        if abs(current_coef[1]) > 0.001:
            yy = -(current_coef[0] * xx + current_intercept) / current_coef[1]
            valid = (yy >= y_min) & (yy <= y_max)
            line_color = '#FFD700' if step < n_steps else '#32CD32'
            ax.plot(xx[valid], yy[valid], color=line_color, linewidth=3,
                   label='Decision Boundary')

        # Add probability contours for final step
        if step == n_steps:
            xx_grid, yy_grid = np.meshgrid(np.linspace(x_min, x_max, 100),
                                           np.linspace(y_min, y_max, 100))
            Z = 1 / (1 + np.exp(-(current_coef[0] * xx_grid + current_coef[1] * yy_grid + current_intercept)))
            ax.contourf(xx_grid, yy_grid, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['#FF6B6B', '#4ECDC4'])

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Logistic Regression - Iteration {step}/{n_steps}', fontsize=14, fontweight='bold')

        # Legend
        died_patch = mpatches.Patch(color='#FF6B6B', label='Died')
        survived_patch = mpatches.Patch(color='#4ECDC4', label='Survived')
        ax.legend(handles=[died_patch, survived_patch], loc='upper right')

        placeholder.pyplot(fig)
        progress_bar.progress((step + 1) / (n_steps + 1))
        status_text.markdown(f"**Iteration {step}**: Optimizing decision boundary...")

        plt.close(fig)
        time.sleep(0.4)

    status_text.success("Optimization complete! Decision boundary found.")

    st.markdown("""
    **What you saw:**
    - The decision boundary (line) separates the two classes
    - It adjusts to minimize classification errors
    - Shaded regions show predicted class probabilities
    - Points on the "wrong" side are misclassified
    """)


# ============================================================================
# SVM VISUALIZATION
# ============================================================================

def visualize_svm(model, X_train, y_train, feature_names, C, kernel):
    """
    Animated SVM showing hyperplane, margin, and support vectors.
    Improved visualization with clearer support vector highlighting and margin display.
    """
    st.markdown("#### SVM Training Animation")
    st.caption("Watch the optimal hyperplane and maximum margin form")

    # Convert to numpy arrays if needed
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Reduce to 2D using PCA
    X_2d, xlabel, ylabel = reduce_to_2d(X_train_np, feature_names)

    # Retrain SVM on 2D data for visualization
    from sklearn.svm import SVC
    svm_2d = SVC(C=C, kernel=kernel, random_state=42)
    svm_2d.fit(X_2d, y_train_np)

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Get plot bounds with padding
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    # Get support vector info
    support_vector_indices = svm_2d.support_
    n_sv = len(support_vector_indices)
    sv_ratio = n_sv / len(y_train_np) * 100

    # Colors
    COLOR_DIED = '#FF6B6B'
    COLOR_SURVIVED = '#4ECDC4'
    COLOR_SV_HIGHLIGHT = '#FFD700'  # Gold for support vectors

    phases = ['data', 'hyperplanes', 'optimal', 'margin', 'support_vectors', 'final']

    for phase_idx, phase in enumerate(phases):
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)

        ax_main = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])

        # === MAIN PLOT ===
        # Create meshgrid for decision boundary
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                            np.linspace(y_min, y_max, 300))

        # Get decision function values
        if phase != 'data':
            Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

        # Draw based on phase
        if phase == 'data':
            # Just show data points
            pass

        elif phase == 'hyperplanes':
            # Show multiple candidate hyperplanes (gray dashed lines)
            for offset in np.linspace(-2, 2, 9):
                ax_main.contour(xx, yy, Z, levels=[offset], colors='gray',
                              linestyles='--', alpha=0.4, linewidths=1)

        elif phase == 'optimal':
            # Show only the optimal hyperplane (decision boundary at Z=0)
            ax_main.contour(xx, yy, Z, levels=[0], colors='black',
                          linestyles='-', linewidths=3)

        elif phase == 'margin':
            # Show hyperplane with margin bands
            # Fill the margin region
            ax_main.contourf(xx, yy, Z, levels=[-1, 1], colors=['#FFFACD'], alpha=0.5)
            # Draw margin boundaries
            ax_main.contour(xx, yy, Z, levels=[-1], colors=[COLOR_DIED],
                          linestyles='--', linewidths=2)
            ax_main.contour(xx, yy, Z, levels=[1], colors=[COLOR_SURVIVED],
                          linestyles='--', linewidths=2)
            # Decision boundary
            ax_main.contour(xx, yy, Z, levels=[0], colors='black',
                          linestyles='-', linewidths=3)

        elif phase in ['support_vectors', 'final']:
            # Full visualization with colored regions
            ax_main.contourf(xx, yy, Z, levels=[-100, 0], colors=[COLOR_DIED], alpha=0.15)
            ax_main.contourf(xx, yy, Z, levels=[0, 100], colors=[COLOR_SURVIVED], alpha=0.15)
            # Margin band
            ax_main.contourf(xx, yy, Z, levels=[-1, 1], colors=['#FFFACD'], alpha=0.3)
            # Margin boundaries
            ax_main.contour(xx, yy, Z, levels=[-1], colors=[COLOR_DIED],
                          linestyles='--', linewidths=2)
            ax_main.contour(xx, yy, Z, levels=[1], colors=[COLOR_SURVIVED],
                          linestyles='--', linewidths=2)
            # Decision boundary
            ax_main.contour(xx, yy, Z, levels=[0], colors='black',
                          linestyles='-', linewidths=3)

        # Plot data points (always)
        for i in range(len(X_2d)):
            x, y = X_2d[i]
            label = y_train_np[i]
            color = COLOR_DIED if label == 0 else COLOR_SURVIVED
            is_sv = i in support_vector_indices

            if phase in ['support_vectors', 'final'] and is_sv:
                # Draw yellow ring around support vectors
                ax_main.scatter(x, y, c='none', s=200, edgecolors=COLOR_SV_HIGHLIGHT,
                              linewidth=3, zorder=4)
                ax_main.scatter(x, y, c=color, s=80, alpha=0.9,
                              edgecolors='black', linewidth=1, zorder=5)
            else:
                alpha = 0.5 if phase in ['support_vectors', 'final'] and not is_sv else 0.7
                ax_main.scatter(x, y, c=color, s=60, alpha=alpha,
                              edgecolors='black', linewidth=0.5, zorder=3)

        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.set_xlabel(xlabel, fontsize=12)
        ax_main.set_ylabel(ylabel, fontsize=12)
        ax_main.set_title(f'SVM ({kernel} kernel) - C={C}', fontsize=14, fontweight='bold')

        # Legend for main plot
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DIED,
                   markersize=10, label='Died (0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SURVIVED,
                   markersize=10, label='Survived (1)'),
        ]
        if phase in ['support_vectors', 'final']:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=10, markeredgecolor=COLOR_SV_HIGHLIGHT,
                       markeredgewidth=3, label='Support Vector')
            )
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # === INFO PANEL ===
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')

        # Title
        ax_info.text(0.5, 0.95, 'SVM Concepts', ha='center', fontsize=14, fontweight='bold')

        if phase == 'data':
            ax_info.text(0.5, 0.75, '📊 Training Data', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.65, f'{len(y_train_np)} samples\n'
                        f'{sum(y_train_np==0)} Died, {sum(y_train_np==1)} Survived',
                        ha='center', fontsize=10, va='top')
            ax_info.text(0.5, 0.45, '🎯 Goal', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.35, 'Find hyperplane that\nbest separates the classes\nwith maximum margin',
                        ha='center', fontsize=10, va='top')

        elif phase == 'hyperplanes':
            ax_info.text(0.5, 0.75, '🔍 Candidate Hyperplanes', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.60, 'Many hyperplanes can\nseparate the data...\n\nBut which one is best?',
                        ha='center', fontsize=10, va='top')
            ax_info.text(0.5, 0.35, '💡 SVM Approach', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.20, 'Choose the one with\nMAXIMUM MARGIN\n(widest gap between classes)',
                        ha='center', fontsize=10, va='top')

        elif phase == 'optimal':
            ax_info.text(0.5, 0.75, '✓ Optimal Hyperplane', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.60, 'The black line is the\ndecision boundary\n\nPoints above → Survived\nPoints below → Died',
                        ha='center', fontsize=10, va='top')

        elif phase == 'margin':
            ax_info.text(0.5, 0.80, '📏 The Margin', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.65, 'Yellow band = Margin\n\nDashed lines are the\nmargin boundaries\nat distance 1 from\nthe hyperplane',
                        ha='center', fontsize=10, va='top')
            ax_info.text(0.5, 0.35, f'⚙️ C = {C}', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.22, 'Higher C → Narrower margin\n(stricter, less violations)\n\nLower C → Wider margin\n(more tolerant)',
                        ha='center', fontsize=9, va='top')

        elif phase == 'support_vectors':
            ax_info.text(0.5, 0.85, '⭐ Support Vectors', ha='center', fontsize=12, fontweight='bold')
            ax_info.text(0.5, 0.72, 'Yellow-ringed points\nare SUPPORT VECTORS',
                        ha='center', fontsize=10, va='top')
            ax_info.text(0.5, 0.55, f'Found: {n_sv} SVs\n({sv_ratio:.1f}% of data)',
                        ha='center', fontsize=11, fontweight='bold')
            ax_info.text(0.5, 0.38, 'These points:\n• Lie on/near margin\n• Define the boundary\n• Are the "critical" points',
                        ha='center', fontsize=9, va='top')
            if sv_ratio > 50:
                ax_info.text(0.5, 0.12, '⚠️ Many SVs indicates\noverlapping classes\n(hard to separate)',
                            ha='center', fontsize=9, color='#CC6600', va='top')

        elif phase == 'final':
            ax_info.text(0.5, 0.85, '✅ SVM Complete!', ha='center', fontsize=12, fontweight='bold',
                        color='green')

            # Summary box
            summary = f"""Kernel: {kernel}
Regularization (C): {C}
Support Vectors: {n_sv}
SV Ratio: {sv_ratio:.1f}%

Decision Rule:
• If f(x) > 0 → Survived
• If f(x) < 0 → Died
• f(x) = 0 is the boundary"""
            ax_info.text(0.5, 0.45, summary, ha='center', fontsize=9, va='center',
                        family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#333'))

        plt.tight_layout()
        placeholder.pyplot(fig)
        progress_bar.progress((phase_idx + 1) / len(phases))

        # Status messages
        status_messages = {
            'data': "**Step 1**: Displaying training data in 2D (PCA projection)...",
            'hyperplanes': "**Step 2**: Considering many possible hyperplanes...",
            'optimal': "**Step 3**: Found optimal hyperplane (maximum margin)!",
            'margin': "**Step 4**: Showing the margin band...",
            'support_vectors': "**Step 5**: Highlighting support vectors...",
            'final': "SVM training visualization complete!"
        }
        if phase == 'final':
            status_text.success(status_messages[phase])
        else:
            status_text.markdown(status_messages[phase])

        plt.close(fig)
        time.sleep(0.8)

    # Final explanation
    st.markdown(f"""
    **What you saw:**
    - **Hyperplane**: The black line separating classes (decision boundary)
    - **Margin**: The yellow band - SVM maximizes this gap
    - **Support Vectors**: {n_sv} points ({sv_ratio:.1f}%) marked with gold rings
    - These critical points "support" the hyperplane position

    **Why so many support vectors?**
    {"With overlapping/non-separable data like Titanic, many points fall within or violate the margin. This is normal!" if sv_ratio > 40 else "The classes are relatively separable in this projection."}
    """)


# ============================================================================
# PERCEPTRON VISUALIZATION
# ============================================================================

def visualize_perceptron(model, X_train, y_train, feature_names, max_iter, eta0):
    """
    Animated perceptron showing weight updates.
    """
    st.markdown("#### Perceptron Training Animation")
    st.caption("Watch the single neuron adjust its weights")

    # Convert to numpy arrays if needed (fixes pandas indexing issues)
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Reduce to 2D
    X_2d, xlabel, ylabel = reduce_to_2d(X_train_np, feature_names)

    # Simulate perceptron learning
    np.random.seed(42)
    weights = np.random.randn(2) * 0.5
    bias = np.random.randn() * 0.5

    # Train to get final weights
    from sklearn.linear_model import Perceptron
    perc_2d = Perceptron(max_iter=max_iter, eta0=eta0, random_state=42)
    perc_2d.fit(X_2d, y_train_np)
    final_weights = perc_2d.coef_[0]
    final_bias = perc_2d.intercept_[0]

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    n_steps = 10
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    for step in range(n_steps + 1):
        t = step / n_steps
        current_weights = weights + t * (final_weights - weights)
        current_bias = bias + t * (final_bias - bias)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Decision boundary
        colors = ['#FF6B6B' if y == 0 else '#4ECDC4' for y in y_train_np]
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

        # Plot decision boundary
        xx = np.linspace(x_min, x_max, 100)
        if abs(current_weights[1]) > 0.001:
            yy = -(current_weights[0] * xx + current_bias) / current_weights[1]
            valid = (yy >= y_min) & (yy <= y_max)
            line_color = '#FFD700' if step < n_steps else '#32CD32'
            ax1.plot(xx[valid], yy[valid], color=line_color, linewidth=3)

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title('Decision Boundary', fontsize=12, fontweight='bold')

        # Right plot: Neuron diagram
        ax2.set_xlim(-2, 4)
        ax2.set_ylim(-2, 2)
        ax2.axis('off')
        ax2.set_title('Perceptron Neuron', fontsize=12, fontweight='bold')

        # Draw inputs
        input_y = [0.8, -0.8]
        for i, y_pos in enumerate(input_y):
            ax2.add_patch(Circle((-1, y_pos), 0.3, facecolor='lightblue', edgecolor='black'))
            ax2.text(-1, y_pos, f'x{i+1}', ha='center', va='center', fontsize=10)

            # Draw weight arrow
            weight_val = current_weights[i]
            color = 'green' if weight_val > 0 else 'red'
            ax2.annotate('', xy=(1.2, 0), xytext=(-0.7, y_pos),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax2.text(0.2, y_pos/2, f'w{i+1}={weight_val:.2f}', fontsize=9, color=color)

        # Draw neuron
        ax2.add_patch(Circle((2, 0), 0.5, facecolor='yellow', edgecolor='black', linewidth=2))
        ax2.text(2, 0, 'f(x)', ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw output
        ax2.annotate('', xy=(3.5, 0), xytext=(2.5, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax2.text(3.7, 0, 'Output', fontsize=10)

        # Show bias
        ax2.text(2, -1.2, f'bias={current_bias:.2f}', ha='center', fontsize=10)

        placeholder.pyplot(fig)
        progress_bar.progress((step + 1) / (n_steps + 1))
        status_text.markdown(f"**Iteration {step}**: Adjusting weights...")

        plt.close(fig)
        time.sleep(0.5)

    status_text.success("Perceptron training complete!")

    st.markdown("""
    **What you saw:**
    - The perceptron is a single artificial neuron
    - Inputs are multiplied by weights and summed
    - The decision boundary moves as weights update
    - Green weights = positive influence, Red = negative
    """)


# ============================================================================
# NEURAL NETWORK VISUALIZATION
# ============================================================================

def visualize_neural_network(model, X_train, y_train, hidden_layer_sizes, activation, max_iter):
    """
    Animated neural network showing layers and activations.
    TensorFlow Playground-style visualization with colored weights.
    """
    st.markdown("#### Neural Network Training Animation")
    st.caption("Watch data flow through the network layers")

    n_features = X_train.shape[1]
    n_outputs = 1

    # Build layer sizes
    if isinstance(hidden_layer_sizes, int):
        hidden_layer_sizes = (hidden_layer_sizes,)
    layer_sizes = [n_features] + list(hidden_layer_sizes) + [n_outputs]

    # Limit display neurons for visualization (show max 6 per layer, with "..." indicator)
    max_display_neurons = 6

    # Get actual weights from the trained model if available
    try:
        weights = model.coefs_
        biases = model.intercepts_
    except:
        # Generate random weights for visualization if model doesn't have them
        np.random.seed(42)
        weights = []
        for i in range(len(layer_sizes) - 1):
            weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5)
        biases = [np.random.randn(s) * 0.1 for s in layer_sizes[1:]]

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Animation phases
    phases = ['architecture'] + [f'forward_{i}' for i in range(len(layer_sizes))] + ['loss_update', 'backward', 'complete']

    # Simulate loss decreasing
    np.random.seed(42)
    losses = [0.7 - i * 0.04 + np.random.uniform(-0.02, 0.02) for i in range(len(phases))]

    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_layer_sizes))] + ['Output']

    for phase_idx, phase in enumerate(phases):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2.5, 1]})
        ax1, ax2 = axes

        # === LEFT PLOT: Network Architecture (TensorFlow Playground style) ===
        ax1.set_xlim(-1, len(layer_sizes) * 2.5)

        # Calculate y limits based on max neurons to display
        max_neurons_display = min(max(layer_sizes), max_display_neurons)
        y_range = max_neurons_display + 2
        ax1.set_ylim(-y_range/2, y_range/2)
        ax1.axis('off')
        ax1.set_title('Network Architecture', fontsize=14, fontweight='bold', pad=20)

        # Determine which layer is "active"
        active_layer = -1
        if phase.startswith('forward_'):
            active_layer = int(phase.split('_')[1])
        elif phase == 'backward':
            active_layer = len(layer_sizes) - 1

        # Store neuron positions for drawing connections
        neuron_positions = []

        # Draw neurons for each layer
        for layer_idx, n_neurons in enumerate(layer_sizes):
            x_pos = layer_idx * 2.5

            # Limit displayed neurons
            display_neurons = min(n_neurons, max_display_neurons)

            # Calculate y positions (centered)
            if display_neurons == 1:
                y_positions = [0]
            else:
                y_positions = np.linspace(-(display_neurons-1)/2 * 0.8, (display_neurons-1)/2 * 0.8, display_neurons)

            layer_positions = []

            for i, y in enumerate(y_positions):
                # Determine neuron color based on activation state
                if layer_idx < active_layer:
                    # Already processed - show activation colors (orange/blue gradient)
                    activation_val = np.random.uniform(-1, 1)
                    if activation_val > 0:
                        color = plt.cm.Oranges(0.3 + abs(activation_val) * 0.5)
                    else:
                        color = plt.cm.Blues(0.3 + abs(activation_val) * 0.5)
                elif layer_idx == active_layer:
                    color = '#FFD700'  # Currently active - gold
                else:
                    color = 'white'  # Not yet processed

                # Draw neuron as circle
                neuron_radius = 0.25
                circle = Circle((x_pos, y), neuron_radius,
                               facecolor=color,
                               edgecolor='#333333',
                               linewidth=2,
                               zorder=10)
                ax1.add_patch(circle)
                layer_positions.append((x_pos, y))

            # Add "..." indicator if there are more neurons than displayed
            if n_neurons > max_display_neurons:
                ax1.text(x_pos, -y_range/2 + 0.8, f'⋮\n+{n_neurons - max_display_neurons} more',
                        ha='center', va='center', fontsize=8, color='gray')

            neuron_positions.append(layer_positions)

            # Layer label at bottom
            ax1.text(x_pos, -y_range/2 + 0.3, f'{layer_names[layer_idx]}\n({n_neurons})',
                    ha='center', va='top', fontsize=10, fontweight='bold')

        # Draw connections with weight-based colors (like TensorFlow Playground)
        for layer_idx in range(len(layer_sizes) - 1):
            pos1 = neuron_positions[layer_idx]
            pos2 = neuron_positions[layer_idx + 1]

            # Get weights for this layer
            layer_weights = weights[layer_idx]

            # Normalize weights for color mapping
            w_max = np.max(np.abs(layer_weights)) if np.max(np.abs(layer_weights)) > 0 else 1

            for i, (x1, y1) in enumerate(pos1):
                if i >= layer_weights.shape[0]:
                    continue
                for j, (x2, y2) in enumerate(pos2):
                    if j >= layer_weights.shape[1]:
                        continue

                    w = layer_weights[i, j]
                    w_normalized = w / w_max

                    # Color: orange for positive, blue for negative (TensorFlow Playground style)
                    if w_normalized > 0:
                        color = plt.cm.Oranges(0.3 + abs(w_normalized) * 0.6)
                    else:
                        color = plt.cm.Blues(0.3 + abs(w_normalized) * 0.6)

                    # Line width based on weight magnitude
                    linewidth = 0.5 + abs(w_normalized) * 2.5

                    # Alpha based on phase
                    if phase == 'backward':
                        alpha = 0.9 if layer_idx >= active_layer - 1 else 0.4
                    elif layer_idx < active_layer:
                        alpha = 0.8
                    else:
                        alpha = 0.3

                    # Draw connection line
                    ax1.plot([x1 + 0.25, x2 - 0.25], [y1, y2],
                            color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        # Add color legend for weights
        legend_x = len(layer_sizes) * 2.5 - 0.5
        legend_y = y_range/2 - 0.5
        ax1.text(legend_x, legend_y, 'Weights:', fontsize=9, fontweight='bold', ha='right')
        ax1.plot([legend_x - 1.5, legend_x - 1], [legend_y - 0.4, legend_y - 0.4],
                color=plt.cm.Oranges(0.7), linewidth=3)
        ax1.text(legend_x - 0.9, legend_y - 0.4, '+', fontsize=10, va='center')
        ax1.plot([legend_x - 1.5, legend_x - 1], [legend_y - 0.8, legend_y - 0.8],
                color=plt.cm.Blues(0.7), linewidth=3)
        ax1.text(legend_x - 0.9, legend_y - 0.8, '−', fontsize=12, va='center')

        # === RIGHT PLOT: Loss curve ===
        ax2.set_xlim(0, len(phases))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Training Step', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot loss up to current phase
        x_loss = list(range(phase_idx + 1))
        y_loss = losses[:phase_idx + 1]
        ax2.plot(x_loss, y_loss, '#E94560', linewidth=2.5, marker='o', markersize=6)
        ax2.fill_between(x_loss, y_loss, alpha=0.2, color='#E94560')

        # Current loss annotation
        if len(y_loss) > 0:
            ax2.annotate(f'Loss: {y_loss[-1]:.3f}',
                        xy=(x_loss[-1], y_loss[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        placeholder.pyplot(fig)
        progress_bar.progress((phase_idx + 1) / len(phases))

        if phase == 'architecture':
            status_text.markdown("**Step 1**: Network architecture initialized")
        elif phase.startswith('forward_'):
            layer = int(phase.split('_')[1])
            status_text.markdown(f"**Forward Pass**: Processing {layer_names[layer]} layer")
        elif phase == 'loss_update':
            status_text.markdown("**Loss Calculation**: Computing prediction error")
        elif phase == 'backward':
            status_text.markdown("**Backpropagation**: Updating weights (error flows backward)")
        elif phase == 'complete':
            status_text.success("Training iteration complete!")

        plt.close(fig)
        time.sleep(0.5)

    st.markdown(f"""
    **What you saw:**
    - **Architecture**: {' → '.join([str(s) for s in layer_sizes])} neurons
    - **Connections**: Orange = positive weights, Blue = negative weights (thicker = stronger)
    - **Forward pass**: Data flows left to right, activating neurons
    - **Backpropagation**: Error flows backward, updating weights
    """)


# ============================================================================
# RANDOM FOREST VISUALIZATION
# ============================================================================

def draw_mini_tree(ax, tree, feature_names, x_offset, y_offset, scale=1.0, max_depth_display=3):
    """
    Draw a mini decision tree diagram on the given axes.
    Shows actual tree structure with nodes and splits.
    """
    tree_struct = tree.tree_

    # Colors
    COLOR_SURVIVED = '#4ECDC4'
    COLOR_DIED = '#FF6B6B'
    COLOR_SPLIT = '#87CEEB'

    def get_node_info(node_id):
        """Get information about a node"""
        is_leaf = tree_struct.children_left[node_id] == tree_struct.children_right[node_id]
        if is_leaf:
            values = tree_struct.value[node_id][0]
            prediction = np.argmax(values)
            return {'is_leaf': True, 'prediction': prediction, 'samples': int(tree_struct.n_node_samples[node_id])}
        else:
            feat_idx = tree_struct.feature[node_id]
            threshold = tree_struct.threshold[node_id]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"X{feat_idx}"
            return {'is_leaf': False, 'feature': feat_name[:6], 'threshold': threshold,
                   'samples': int(tree_struct.n_node_samples[node_id])}

    def draw_node(node_id, x, y, depth, x_span):
        """Recursively draw nodes"""
        if depth > max_depth_display or node_id == -1:
            return

        info = get_node_info(node_id)
        node_width = 0.35 * scale
        node_height = 0.2 * scale

        if info['is_leaf']:
            # Leaf node - colored by prediction
            color = COLOR_SURVIVED if info['prediction'] == 1 else COLOR_DIED
            rect = FancyBboxPatch((x - node_width/2 + x_offset, y - node_height/2 + y_offset),
                                 node_width, node_height, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='#333', linewidth=1)
            ax.add_patch(rect)
            label = 'S' if info['prediction'] == 1 else 'D'
            ax.text(x + x_offset, y + y_offset, label, ha='center', va='center',
                   fontsize=7*scale, fontweight='bold', color='white')
        else:
            # Split node
            rect = FancyBboxPatch((x - node_width/2 + x_offset, y - node_height/2 + y_offset),
                                 node_width, node_height, boxstyle="round,pad=0.02",
                                 facecolor=COLOR_SPLIT, edgecolor='#333', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + x_offset, y + y_offset, info['feature'], ha='center', va='center',
                   fontsize=6*scale, fontweight='bold', color='#333')

            # Draw children
            if depth < max_depth_display:
                child_y = y - 0.35 * scale
                left_x = x - x_span/4
                right_x = x + x_span/4

                # Draw edges
                ax.plot([x + x_offset, left_x + x_offset], [y - node_height/2 + y_offset, child_y + node_height/2 + y_offset],
                       'k-', linewidth=0.8, alpha=0.6)
                ax.plot([x + x_offset, right_x + x_offset], [y - node_height/2 + y_offset, child_y + node_height/2 + y_offset],
                       'k-', linewidth=0.8, alpha=0.6)

                # Recurse
                draw_node(tree_struct.children_left[node_id], left_x, child_y, depth + 1, x_span/2)
                draw_node(tree_struct.children_right[node_id], right_x, child_y, depth + 1, x_span/2)

    # Start drawing from root
    draw_node(0, 0, 0.4 * scale, 0, 1.2 * scale)


def visualize_random_forest(model, X_train, y_train, feature_names, n_estimators, max_depth):
    """
    Animated random forest visualization showing actual tree structures.
    """
    st.markdown("#### Random Forest Training Animation")
    st.caption("Watch decision trees being built and combined for voting")

    # Convert to numpy arrays if needed
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Limit trees for visualization (show 6 trees in a 2x3 grid)
    n_viz_trees = min(n_estimators, 6)

    # Sample for voting visualization
    np.random.seed(42)
    n_samples = min(5, len(X_train_np))
    sample_indices = np.random.choice(len(X_train_np), n_samples, replace=False)
    X_sample = X_train_np[sample_indices]
    y_sample = y_train_np[sample_indices]

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Colors
    COLOR_SURVIVED = '#4ECDC4'
    COLOR_DIED = '#FF6B6B'

    # Animation phases: build each tree, then voting
    phases = ['init'] + [f'tree_{i}' for i in range(n_viz_trees)] + ['voting', 'final']

    for phase_idx, phase in enumerate(phases):
        fig = plt.figure(figsize=(18, 10))

        # Layout: Top row = trees (3), Bottom left = trees (3), Bottom right = voting results
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2], height_ratios=[1, 1],
                             hspace=0.4, wspace=0.3)

        # Determine trees built
        if phase == 'init':
            trees_built = 0
        elif phase.startswith('tree_'):
            trees_built = int(phase.split('_')[1]) + 1
        else:
            trees_built = n_viz_trees

        # Draw tree subplots (2 rows x 3 cols)
        tree_axes = []
        for i in range(n_viz_trees):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            tree_axes.append(ax)

            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.6, 0.6)
            ax.axis('off')

            # Title with tree number
            is_current = (phase == f'tree_{i}')
            title_color = '#FFD700' if is_current else '#333'
            title_weight = 'bold'
            ax.set_title(f'Tree {i+1}', fontsize=11, fontweight=title_weight, color=title_color, pad=5)

            if i < trees_built:
                # Draw the actual tree structure
                tree = model.estimators_[i]
                draw_mini_tree(ax, tree, feature_names, x_offset=0, y_offset=0, scale=0.9, max_depth_display=3)

                # Add prediction for sample 1
                pred = tree.predict(X_sample[:1])[0]
                pred_color = COLOR_SURVIVED if pred == 1 else COLOR_DIED
                pred_text = 'Survived' if pred == 1 else 'Died'
                ax.text(0, -0.55, f'Vote: {pred_text}', ha='center', fontsize=8,
                       fontweight='bold', color=pred_color,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=pred_color, linewidth=2))

                # Highlight border if currently building
                if is_current:
                    rect = plt.Rectangle((-0.75, -0.55), 1.5, 1.1, fill=False,
                                        edgecolor='#FFD700', linewidth=4, linestyle='-')
                    ax.add_patch(rect)
            else:
                # Not yet built
                ax.text(0, 0, 'Not built yet', ha='center', va='center',
                       fontsize=10, color='gray', style='italic')
                ax.add_patch(plt.Rectangle((-0.6, -0.4), 1.2, 0.8, fill=False,
                                          edgecolor='#ccc', linewidth=2, linestyle='--'))

        # === RIGHT SIDE: Voting Results Panel ===
        ax_voting = fig.add_subplot(gs[:, 3])
        ax_voting.set_xlim(0, 1)
        ax_voting.set_ylim(0, 1)
        ax_voting.axis('off')
        ax_voting.set_title('Ensemble Voting', fontsize=14, fontweight='bold', pad=10)

        if trees_built > 0 and phase not in ['init']:
            # Calculate votes for sample 1
            votes = [model.estimators_[t].predict(X_sample[:1])[0] for t in range(trees_built)]
            votes_survived = sum(votes)
            votes_died = trees_built - votes_survived

            # Vote breakdown header
            ax_voting.text(0.5, 0.92, f'Sample 1 ({trees_built} trees voting)', ha='center',
                          fontsize=11, fontweight='bold')

            # Visual vote representation - show each tree's vote as a small box
            vote_y = 0.82
            box_size = 0.08
            start_x = 0.5 - (trees_built * box_size * 1.2) / 2

            for i, v in enumerate(votes):
                color = COLOR_SURVIVED if v == 1 else COLOR_DIED
                x = start_x + i * box_size * 1.2
                rect = plt.Rectangle((x, vote_y - box_size/2), box_size, box_size,
                                    facecolor=color, edgecolor='#333', linewidth=1)
                ax_voting.add_patch(rect)
                ax_voting.text(x + box_size/2, vote_y, f'T{i+1}', ha='center', va='center',
                             fontsize=6, color='white', fontweight='bold')

            # Vote count boxes
            ax_voting.add_patch(plt.Rectangle((0.1, 0.55), 0.35, 0.15, facecolor=COLOR_SURVIVED,
                                             edgecolor='#333', linewidth=2))
            ax_voting.text(0.275, 0.625, f'Survived: {votes_survived}', ha='center', va='center',
                          fontsize=11, fontweight='bold', color='white')

            ax_voting.add_patch(plt.Rectangle((0.55, 0.55), 0.35, 0.15, facecolor=COLOR_DIED,
                                             edgecolor='#333', linewidth=2))
            ax_voting.text(0.725, 0.625, f'Died: {votes_died}', ha='center', va='center',
                          fontsize=11, fontweight='bold', color='white')

            # Final prediction
            if phase in ['voting', 'final']:
                final_pred = 'Survived' if votes_survived > votes_died else 'Died'
                final_color = COLOR_SURVIVED if votes_survived > votes_died else COLOR_DIED
                actual = 'Survived' if y_sample[0] == 1 else 'Died'
                is_correct = (final_pred == actual)

                # Arrow pointing to winner
                winner_x = 0.275 if votes_survived > votes_died else 0.725
                ax_voting.annotate('', xy=(winner_x, 0.45), xytext=(winner_x, 0.53),
                                  arrowprops=dict(arrowstyle='->', color='#333', lw=2))

                # Final prediction box
                ax_voting.add_patch(FancyBboxPatch((0.15, 0.2), 0.7, 0.2,
                                                  boxstyle="round,pad=0.03",
                                                  facecolor=final_color, edgecolor='#333', linewidth=3))
                ax_voting.text(0.5, 0.3, f'Prediction: {final_pred}', ha='center', va='center',
                              fontsize=13, fontweight='bold', color='white')

                # Correctness
                result_text = '✓ Correct!' if is_correct else '✗ Incorrect'
                result_color = '#32CD32' if is_correct else '#FF4444'
                ax_voting.text(0.5, 0.08, result_text, ha='center', fontsize=12,
                              fontweight='bold', color=result_color)
                ax_voting.text(0.5, 0.02, f'(Actual: {actual})', ha='center', fontsize=9, color='gray')
            else:
                ax_voting.text(0.5, 0.3, 'Building more trees...', ha='center', va='center',
                              fontsize=11, color='gray', style='italic')
        else:
            ax_voting.text(0.5, 0.5, 'Waiting for trees\nto be built...', ha='center', va='center',
                          fontsize=12, color='gray', style='italic')

        # Legend at bottom
        legend_elements = [
            FancyBboxPatch((0, 0), 1, 1, boxstyle="round", facecolor='#87CEEB', edgecolor='#333', label='Split Node'),
            FancyBboxPatch((0, 0), 1, 1, boxstyle="round", facecolor=COLOR_SURVIVED, edgecolor='#333', label='Leaf: Survived'),
            FancyBboxPatch((0, 0), 1, 1, boxstyle="round", facecolor=COLOR_DIED, edgecolor='#333', label='Leaf: Died'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
                  bbox_to_anchor=(0.4, 0.01))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        placeholder.pyplot(fig)
        progress_bar.progress((phase_idx + 1) / len(phases))

        # Status messages
        if phase == 'init':
            status_text.markdown("**Initializing**: Preparing to build random forest...")
        elif phase.startswith('tree_'):
            tree_num = int(phase.split('_')[1]) + 1
            status_text.markdown(f"**Building Tree {tree_num}/{n_viz_trees}**: Training on bootstrap sample with random feature subset")
        elif phase == 'voting':
            status_text.markdown("**Voting Phase**: All trees cast their votes based on learned patterns...")
        elif phase == 'final':
            status_text.success(f"Random Forest complete! {n_viz_trees} trees vote → Majority wins!")

        plt.close(fig)
        time.sleep(0.6 if phase.startswith('tree_') else 1.0)

    st.markdown(f"""
    **What you saw:**
    - **{n_viz_trees} decision trees** built from the forest
    - Each tree learns different patterns (bootstrap sampling + random features)
    - **Blue nodes** = decision splits, **Teal** = predict Survived, **Coral** = predict Died
    - Trees vote independently → **Majority vote** = final prediction
    - Ensemble typically more accurate than any single tree!
    """)


# ============================================================================
# RIDGE REGRESSION VISUALIZATION
# ============================================================================

def visualize_ridge_regression(model, X_train, y_train, feature_names, alpha):
    """
    Animated ridge regression showing coefficient shrinkage.
    """
    st.markdown("#### Ridge Regression (L2) Animation")
    st.caption("Watch how regularization shrinks coefficients toward zero")

    # Convert to numpy arrays if needed (fixes pandas indexing issues)
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Get OLS coefficients for comparison
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_train_np, y_train_np)
    ols_coefs = ols.coef_

    ridge_coefs = model.coef_

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    n_steps = 10

    for step in range(n_steps + 1):
        t = step / n_steps
        current_coefs = ols_coefs + t * (ridge_coefs - ols_coefs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Coefficient bar chart
        x_pos = np.arange(len(feature_names))
        width = 0.35

        ax1.bar(x_pos - width/2, ols_coefs, width, label='OLS (no regularization)', color='#FF6B6B', alpha=0.7)
        ax1.bar(x_pos + width/2, current_coefs, width, label=f'Ridge (alpha={alpha})', color='#4ECDC4', alpha=0.7)

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Coefficient Value', fontsize=12)
        ax1.set_title('Coefficient Comparison', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.legend()

        # Right plot: L2 penalty visualization
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)

        # Draw L2 constraint circle
        circle = plt.Circle((0, 0), 1, fill=False, color='blue', linewidth=2, linestyle='--')
        ax2.add_patch(circle)
        ax2.text(0, -1.3, 'L2 Constraint\n(sum of squared coefs)', ha='center', fontsize=9)

        # Plot coefficient path (simplified 2D)
        if len(current_coefs) >= 2:
            # Normalize for visualization
            scale = max(np.max(np.abs(ols_coefs[:2])), 1)
            ols_point = ols_coefs[:2] / scale
            current_point = current_coefs[:2] / scale

            ax2.scatter(*ols_point, c='red', s=100, marker='x', label='OLS', zorder=5)
            ax2.scatter(*current_point, c='green', s=100, marker='o', label='Ridge', zorder=5)
            ax2.plot([ols_point[0], current_point[0]], [ols_point[1], current_point[1]],
                    'g--', alpha=0.5, linewidth=2)

        ax2.set_xlabel(f'{feature_names[0]} coefficient', fontsize=10)
        ax2.set_ylabel(f'{feature_names[1]} coefficient', fontsize=10)
        ax2.set_title('L2 Regularization Effect', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        placeholder.pyplot(fig)
        progress_bar.progress((step + 1) / (n_steps + 1))
        status_text.markdown(f"**Step {step}**: Applying L2 penalty (alpha={alpha})...")

        plt.close(fig)
        time.sleep(0.4)

    status_text.success("Ridge regression complete!")

    # Calculate shrinkage
    shrinkage = np.mean(np.abs(ridge_coefs) / (np.abs(ols_coefs) + 1e-10))
    st.markdown(f"""
    **What you saw:**
    - **OLS coefficients** (red) have no constraints
    - **Ridge coefficients** (green) are **shrunk toward zero**
    - L2 penalty adds constraint: sum of squared coefficients
    - Average shrinkage: **{(1-shrinkage)*100:.1f}%** reduction in coefficient magnitude
    """)


# ============================================================================
# LASSO REGRESSION VISUALIZATION
# ============================================================================

def visualize_lasso_regression(model, X_train, y_train, feature_names, alpha):
    """
    Animated lasso regression showing feature selection (coefficients going to zero).
    """
    st.markdown("#### Lasso Regression (L1) Animation")
    st.caption("Watch how L1 regularization eliminates features (coefficients become exactly zero)")

    # Convert to numpy arrays if needed (fixes pandas indexing issues)
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # Get OLS coefficients for comparison
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_train_np, y_train_np)
    ols_coefs = ols.coef_

    lasso_coefs = model.coef_

    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    n_steps = 10

    for step in range(n_steps + 1):
        t = step / n_steps
        current_coefs = ols_coefs + t * (lasso_coefs - ols_coefs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Coefficient bar chart with zero highlighting
        x_pos = np.arange(len(feature_names))

        colors = []
        for c in current_coefs:
            if abs(c) < 0.001:
                colors.append('#FF0000')  # Red for zero
            else:
                colors.append('#4ECDC4')  # Teal for non-zero

        ax1.bar(x_pos, current_coefs, color=colors, alpha=0.8, edgecolor='black')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Coefficient Value', fontsize=12)
        ax1.set_title(f'Lasso Coefficients (alpha={alpha})', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # Count zeros
        n_zeros = np.sum(np.abs(current_coefs) < 0.001)
        ax1.text(0.02, 0.98, f'Zero coefficients: {n_zeros}/{len(current_coefs)}',
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Right plot: L1 penalty visualization (diamond)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)

        # Draw L1 constraint diamond
        diamond = plt.Polygon([(-1, 0), (0, 1), (1, 0), (0, -1)],
                             fill=False, color='blue', linewidth=2, linestyle='--')
        ax2.add_patch(diamond)
        ax2.text(0, -1.5, 'L1 Constraint\n(sum of |coefs|)', ha='center', fontsize=9)

        # Plot coefficient path
        if len(current_coefs) >= 2:
            scale = max(np.max(np.abs(ols_coefs[:2])), 1)
            ols_point = ols_coefs[:2] / scale
            current_point = current_coefs[:2] / scale

            ax2.scatter(*ols_point, c='red', s=100, marker='x', label='OLS', zorder=5)
            ax2.scatter(*current_point, c='green', s=100, marker='o', label='Lasso', zorder=5)
            ax2.plot([ols_point[0], current_point[0]], [ols_point[1], current_point[1]],
                    'g--', alpha=0.5, linewidth=2)

        # Highlight axes (where coefficients hit zero)
        ax2.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.3)
        ax2.axvline(x=0, color='red', linestyle='-', linewidth=2, alpha=0.3)
        ax2.text(1.5, 0.1, 'Coef 2 = 0', fontsize=8, color='red')
        ax2.text(0.1, 1.5, 'Coef 1 = 0', fontsize=8, color='red')

        ax2.set_xlabel(f'{feature_names[0]} coefficient', fontsize=10)
        ax2.set_ylabel(f'{feature_names[1]} coefficient', fontsize=10)
        ax2.set_title('L1 Regularization Effect', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.set_aspect('equal')

        placeholder.pyplot(fig)
        progress_bar.progress((step + 1) / (n_steps + 1))
        status_text.markdown(f"**Step {step}**: Applying L1 penalty...")

        plt.close(fig)
        time.sleep(0.4)

    status_text.success("Lasso regression complete!")

    # List eliminated features
    zero_features = [f for f, c in zip(feature_names, lasso_coefs) if abs(c) < 0.001]
    kept_features = [f for f, c in zip(feature_names, lasso_coefs) if abs(c) >= 0.001]

    st.markdown(f"""
    **What you saw:**
    - L1 penalty pushes coefficients to **exactly zero** (red bars)
    - This performs **automatic feature selection**
    - The diamond constraint hits axes, forcing zeros

    **Feature Selection Result:**
    - **Eliminated**: {', '.join(zero_features) if zero_features else 'None'}
    - **Kept**: {', '.join(kept_features) if kept_features else 'None'}
    """)
