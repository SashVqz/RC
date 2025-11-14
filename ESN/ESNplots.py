import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ESNmodel import ESNPipeline

def internalStatesHistogram(states, time_step_index, reservoir_size, filename=None):
    if states is None or states.shape[1] == 0:
        return
    if not (0 <= time_step_index < states.shape[1]):
        time_step_index = 0

    states_at_t = states[:, time_step_index]

    plt.figure(figsize=(4, 6))
    plt.hist(states_at_t, bins='auto', color='dodgerblue', edgecolor='black')
    plt.title(f't={time_step_index}')
    plt.xlabel('Internal States Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def predictionAnalysis(predictions, actuals, filename=None, zoom_limit=500):
    if predictions is None or actuals is None or len(predictions) == 0:
        return

    if len(predictions) != len(actuals):
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        if min_len == 0:
            return

    time_steps = np.arange(len(actuals))
    absolute_error = np.abs(actuals - predictions)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)


    zoom_actual = min(zoom_limit, len(actuals))
    axes[0].plot(time_steps[:zoom_actual], actuals[:zoom_actual], label='Target signal', color='steelblue', alpha=0.8)
    axes[0].plot(time_steps[:zoom_actual], predictions[:zoom_actual], label='Free-running predicted signal', color='mediumpurple', linestyle='--', alpha=0.9)
    axes[0].set_ylabel('y(t)')
    axes[0].set_xlabel(f'Prediction Time Step (First {zoom_actual})')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(time_steps, absolute_error, label='|Δ|', color='firebrick', alpha=0.8)
    axes[1].set_ylabel('|Δ|')
    axes[1].set_xlabel('Prediction Time Step')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout(pad=2.0)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
