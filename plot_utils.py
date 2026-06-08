import numpy as np
import matplotlib.pyplot as plt
import itertools


def _nanmean_or_nan(values):
    valid_values = [value for value in values if not np.isnan(value)]
    return np.mean(valid_values) if valid_values else np.nan


def plot_sliding_accuracy(
    model_results,
    window=100,
    show_inline=True,
    save_path=None,
    event_label="Drift",
    known_drift=None
):
    """
    Plots sliding window accuracy for different models.
    
    :param model_results: dict mapping model_name to a dictionary containing:
        - 'accuracy_series': list of 100 (correct) or 0 (wrong)
        - 'drifts': list of instance indices where drift occurred
        - 'overall_accuracy': float representing overall accuracy percentage (optional, computed if missing)
    :param window: Sliding window size.
    :param show_inline: Whether to call plt.show()
    :param save_path: Optional path to save the plot.
    :param event_label: Label used for vertical event markers.
    """
    colors = itertools.cycle(plt.cm.Dark2.colors) 
    marker = itertools.cycle(('+', 'x', 'o', '*'))
    
    plt.figure(figsize=(15, 8))
    plt.xlabel('Instances')
    plt.ylabel('Accuracy [%]')
    total_len = max([len(d.get('accuracy_series', [])) for d in model_results.values()]) if model_results else 0
    if known_drift is not None and total_len > 0:
        if isinstance(known_drift, (int, float)):
            drift_indices = np.arange(known_drift, total_len, known_drift)
        else:
            drift_indices = known_drift

        for i, drift_idx in enumerate(drift_indices):
            label = "KNOWN DRIFT" if i == 0 else ""
            plt.axvline(
                x=drift_idx, 
                color='black', 
                linestyle='-', 
                linewidth=2, 
                label=label, 
                alpha=0.8,
                zorder=1
            )
    
    for model_name, data in model_results.items():
        accuracy_series = data.get('accuracy_series', [])
        drifts = data.get('drifts', [])
        overall_acc = data.get('overall_accuracy')
        if overall_acc is None:
            valid_acc = [a for a in accuracy_series if not np.isnan(a)]
            overall_acc = np.mean(valid_acc) if valid_acc else 0.0
            
        sliding_window_accuracy = []
        for ind in range(len(accuracy_series)):
            window_start = max(0, ind - window + 1)
            sliding_window_accuracy.append(_nanmean_or_nan(accuracy_series[window_start:ind + 1]))

        instance_indexes = np.arange(len(accuracy_series))
        
        current_color = next(colors)
        label = f"{model_name} (Overall: {overall_acc:.1f}%)"
        
        plt.plot(instance_indexes[::window], sliding_window_accuracy[::window], color=current_color,
                 label=label, marker=next(marker), markersize=8, alpha=0.5)
                 
        for i, drift_idx in enumerate(drifts):
            drift_label = f'{model_name} {event_label}' if i == 0 else ""            
            plt.axvline(x=drift_idx - 1, color=current_color, linestyle='--', alpha=0.5, label=drift_label)
            
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    if show_inline:
        plt.show()
