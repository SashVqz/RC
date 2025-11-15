import numpy as np

def powerlawNoiseGenerator(N, beta, randomSeed=None):
    """
    algorithm to generate 1/f^beta noise using the Fourier filtering method. Timmer and Koenig (1995)
    proposed this method to create synthetic light curves with a specified power spectral density.
    """
    if randomSeed is not None:
        np.random.seed(randomSeed)

    N_unique = N // 2 + 1
    frequencies = np.fft.rfftfreq(N)[1:] # 1 to N/2
    
    # (P ~ 1 / f^beta)
    power_spectrum = frequencies**(-beta)
    amplitudes = np.sqrt(power_spectrum)
    phases = np.random.uniform(0, 2 * np.pi, N_unique - 1)
    
    # f_coeffs = A * e^(i*phi) = A * (cos(phi) + i*sin(phi))
    fourier_coeffs_positive = amplitudes * (np.cos(phases) + 1j * np.sin(phases))    
    fourier_coeffs_full = np.concatenate(([0+0j], fourier_coeffs_positive))

    # irfft thats fourier coeffs to time series
    time_series = np.fft.irfft(fourier_coeffs_full, n=N)    
    time_series_normalized = (time_series - np.mean(time_series)) / np.std(time_series)
    
    return time_series_normalized


def mackeyGlassGenerator(length, tau=17, delta_t=1, beta=0.2, gamma=0.1, n=10, x0=1.2):
    """
    Generate Mackey-Glass time series using Euler method for numerical integration.
    """
    x = np.zeros(length)
    x[0] = x0
    for t in range(1, length):
        if t - tau >= 0:
            x_tau = x[t - tau]
        else:
            x_tau = 0.0
        dx_dt = beta * x_tau / (1 + x_tau**n) - gamma * x[t - 1]
        x[t] = x[t - 1] + dx_dt * delta_t
    return x

