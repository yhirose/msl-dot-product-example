MSL Dot Product example
=======================

This library uses GPU cores in Apple M1 chip with [Metal-cpp](https://developer.apple.com/metal/cpp/)

Build and run unit tests and benchmark
--------------------------------------

 * Install Xcode Command Line Tools
 * Run the following commands in Terminal

```bash
cd test
make
```

Benchmark on M1 MacBook Pro 14
------------------------------

Matrix size: [100 × 1000] `dot` [1000 × 100] = [1000 × 100]

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      152,126,173.67 |                6.57 |    0.0% |     18.29 | `CPU`
|        3,717,496.50 |              269.00 |    0.4% |      0.45 | `SIMD w/ Eigen`
|        1,410,621.50 |              708.91 |   11.8% |      0.16 | `GPU`

MSL code for Dot Product
-------------------------

```cpp
kernel void dot(
  device const void* A_bytes,
  device const void* B_bytes,
  device void*       OUT_bytes,
  constant uint32_t& A_cols,
  constant uint32_t& OUT_raws,
  constant uint32_t& OUT_cols,
  uint2              gid [[thread_position_in_grid]])
{
  auto A = static_cast<device const float*>(A_bytes);
  auto B = static_cast<device const float*>(B_bytes);
  auto OUT = reinterpret_cast<device float*>(OUT_bytes);

  float val = 0.0;
  for (uint32_t i = 0; i < A_cols; i++) {
    val += A[A_cols * gid.y + i] * B[OUT_cols * i + gid.x];
  }

  OUT[OUT_cols * gid.y + gid.x] = val;
}
```
