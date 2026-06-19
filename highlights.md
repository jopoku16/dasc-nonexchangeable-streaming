# Highlights

- Introduces DASC, a drift-aware spectral conformal method for streaming non-exchangeable data.
- Provides a diagnostic triangle linking coverage error, transport drift, and effective sample size.
- Defines a DASC reliability index for practical monitoring and fallback decisions.
- Establishes a drift-gating bias-variance theorem for weighted conformal calibration.
- Benchmarks DASC against rolling, adaptive, conformal PID, exponentially weighted, and spectral-only conformal baselines.
- Adds external EnbPI and AgACI-style comparisons on the post-drift benchmark.
- Shows that spectral-only and exponentially weighted conformal prediction can fail after drift despite recurring structure.
- Adds a five-scenario stress-test suite covering abrupt shifts, gradual frequency drift, heavy tails, mixed drift, and weak recurrence.
- Demonstrates 28-42% interval-width reductions on electricity and weather streams relative to the best calibrated non-DASC baseline.
