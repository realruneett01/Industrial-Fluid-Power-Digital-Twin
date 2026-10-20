# Inference Latency Profile

## Latency Percentiles

- p50: 13.89 ms
- p95: 15.29 ms
- p99: 15.97 ms
- Cycles profiled: 10

## Methodology

Measured via time.perf_counter_ns for single-cycle feature extraction plus 4-target ONNX runtime inference on CPU.
