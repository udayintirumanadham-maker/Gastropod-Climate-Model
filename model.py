import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================================
# PHASE 1: STOCHASTIC EVOLUTIONARY SIMULATION (STOCHASTIC CALCULUS - SDE)
# =========================================================================
print("Executing Continuous Stochastic Calculus Simulation for Gastropods...")
np.random.seed(42)
n_samples = 300

# 1. Establish the time scale (Millions of Years Ago around a climate spike, 57 to 55 Ma)
time_steps = np.linspace(57.0, 55.0, n_samples)
dt = np.abs(time_steps[1] - time_steps[0])

# 2. Simulate an abrupt hyperthermal event (Ocean Temperature Shock Curve)
true_temperature = 15.0 + 8.0 * np.exp(-((time_steps - 56.0) / 0.3)**2)
observed_temperature = true_temperature + np.random.normal(0, 0.4, n_samples)

# 3. Model evolution via an Ornstein-Uhlenbeck SDE
# Baseline log-size drops linearly as temperature spikes (The Lilliput Effect Rule)
optimal_log_size = 3.5 - 0.12 * (true_temperature - 15.0)

# Run the Euler-Maruyama solver for the SDE
log_size_evolution = np.zeros(n_samples)
log_size_evolution[0] = optimal_log_size[0]

theta = 4.0      # Speed of evolutionary adaptation back to optimal size
sigma_sde = 0.2  # Volatility / environmental randomness parameter

for t in range(1, n_samples):
    drift = theta * (optimal_log_size[t] - log_size_evolution[t-1]) * dt
    diffusion = sigma_sde * np.sqrt(dt) * np.random.normal()
    log_size_evolution[t] = log_size_evolution[t-1] + drift + diffusion

# Convert back to true physical sizes (millimeters)
fossil_sizes_mm = np.exp(log_size_evolution)

# =========================================================================
# PHASE 2: BAYESIAN INFERENCE ENGINE (PROBABILITY ANALYSIS)
# =========================================================================
print("Running Analytical Bayesian Inference...")

# Feature engineering: Track deviation from baseline temperature
delta_temp = observed_temperature - 15.0
X = delta_temp
Y = log_size_evolution - 3.5  # Centered log-size tracking

# Define Gaussian Priors for our Lilliput Sensitivity Coefficient (Beta)
# Prior: Normal(Mean = 0, Variance = 1) -> Agnostic, unbiased starting view
prior_mean = 0.0
prior_variance = 1.0

# Define integrated observation noise variance from our SDE path
assumed_noise_variance = (sigma_sde**2) * dt

# Apply the Bayesian Conjugate Normal Update Equations
prior_precision = 1.0 / prior_variance
data_precision = np.sum(X**2) / assumed_noise_variance

posterior_variance = 1.0 / (prior_precision + data_precision)
posterior_mean = posterior_variance * ((prior_mean / prior_variance) + (np.sum(X * Y) / assumed_noise_variance))

print("\n=== INFERENCE RESULTS ===")
print(f"Calculated Bayesian Posterior Mean (Beta Value): {posterior_mean:.4f}")
print(f"Calculated Posterior Variance (Uncertainty Bounds): {posterior_variance:.4e}")

# =========================================================================
# PHASE 3: COMPUTATIONAL VISUALIZATION
# =========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top Plot: Temperature Shock Event
ax1.plot(time_steps, observed_temperature, label='Observed Sea Surface Temp (°C)', color='crimson', alpha=0.6)
ax1.plot(time_steps, true_temperature, label='True Climate Trend', color='darkred', linewidth=2)
ax1.set_ylabel('Ocean Temperature (°C)')
ax1.title.set_text('Stochastic Evolutionary Responses of Marine Gastropods to Hyperthermal Climate Change')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Bottom Plot: Evolutionary Fossil Tracking via SDE solver
ax2.scatter(time_steps, fossil_sizes_mm, label='Simulated Gastropod Fossils', color='teal', s=12, alpha=0.5)
ax2.plot(time_steps, np.exp(optimal_log_size), label='Theoretical Optimal Size', color='black', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Geological Time (Millions of Years Ago)')
ax2.set_ylabel('Fossil Body Size Width (mm)')
ax2.invert_xaxis()  # Deep time convention goes backwards into the past
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
