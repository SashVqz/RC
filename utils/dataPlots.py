import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries(series, title=None, max_points=1000, filename=None):
    if not isinstance(series, np.ndarray):
        series = np.array(series)
        
    if series.ndim > 1:
        series_1d = series[:, 0]
    else:
        series_1d = series
        
    series_to_plot = series_1d[:max_points]
    
    plt.figure(figsize=(12, 4))
    plt.plot(series_to_plot)
    plt.title(title, fontsize=14)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_attractor(series, title=None, dim1=0, dim2=1, max_points=5000, filename=None):
    if not isinstance(series, np.ndarray) or series.ndim < 2 or series.shape[1] < max(dim1, dim2) + 1:
        print(f"El plot del atractor requiere al menos {max(dim1, dim2) + 1} dimensiones. Se omite '{title}'.")
        return

    series_to_plot = series[:max_points]

    plt.figure(figsize=(7, 7))
    plt.plot(series_to_plot[:, dim1], series_to_plot[:, dim2], 'o', markersize=0.5, alpha=0.3)
    # plt.title(f"{title} (Var {dim1} Var {dim2})")
    plt.xlabel(f"Var {dim1}")
    plt.ylabel(f"Var {dim2}")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()