# SplineLinearHD Benchmark Results

- **Device:** cuda
- **Samples per dim:** 1000
- **Clusters:** 5
- **Epochs:** 20
- **Learning rate:** 0.001
- **Inference runs:** 100 (batch=32)

## Results Table

| Dim | Dense Params | Dense Acc | Dense Time (ms) | SHD-16 Params | SHD-16 Acc | SHD-16 Ratio | SHD-16 Time (ms) | SHD-32 Params | SHD-32 Acc | SHD-32 Ratio | SHD-32 Time (ms) |
|----:|------------:|----------:|----------------:|-------------:|-----------:|-------------:|-----------------:|-------------:|-----------:|-------------:|-----------------:|
| 64 | 325 | 1.0000 | 0.016 | 85 | 1.0000 | 3.8x | 0.777 | 41 | 0.9900 | 7.9x | 0.396 |
| 128 | 645 | 1.0000 | 0.011 | 197 | 1.0000 | 3.3x | 2.174 | 85 | 1.0000 | 7.6x | 0.796 |
| 256 | 1285 | 1.0000 | 0.014 | 517 | 1.0000 | 2.5x | 2.756 | 197 | 1.0000 | 6.5x | 1.363 |
| 512 | 2565 | 1.0000 | 0.013 | 1541 | 1.0000 | 1.7x | 5.545 | 517 | 1.0000 | 5.0x | 2.874 |
| 768 | 3845 | 1.0000 | 0.021 | 3077 | 1.0000 | 1.2x | 9.315 | 965 | 1.0000 | 4.0x | 4.311 |

## Summary

- **Avg Dense accuracy:** 1.0000
- **Avg SHD-16 accuracy:** 1.0000 (retention: 100.0%)
- **Avg SHD-32 accuracy:** 0.9980 (retention: 99.8%)
- **Avg SHD-16 compression:** 2.5x
- **Avg SHD-32 compression:** 6.2x
