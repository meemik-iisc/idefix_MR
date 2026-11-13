# Single vs double precision — tcgm run (sequential, GPU not shared)

Grid 4,194,304 cells, t_end = 60 code units, GPU run.

| Quantity | Double | Single | Double / Single |
|---|---|---|---|
| Wall time | 1 hour 34 minutes 19 seconds | 41 minutes 4 seconds | 2.30 |
| Wall time [s] | 5659 | 2464 | 2.30 |
| Cycles | 66667 | 66572 | 1.00 |
| Wall time per cycle [ms] | 84.9 | 37.0 | 2.29 |
| Throughput [Mcell-updates/s] | 49.4 | 113.3 | 0.44 |
| Median interval throughput [Mcell/s] | 55.0 | 120.4 | 0.46 |
| Peak GPU memory [MB] | 1195 | 598 | 2.00 |
| Peak host memory [MB] | 549 | 275 | 2.00 |
| Mean VTK write time [s] | 0.094 | 0.047 | 2.01 |
| Mean dump write time [s] | 0.113 | 0.038 | 2.94 |
| Total VTK size [MB] | 5862 | 5862 | 1.00 |
| Total dump size [MB] | 1345 | 673 | 2.00 |
| Median dt | 8.761e-04 | 8.452e-04 | 1.04 |
| Median cooling nsub | 449 | 482 | 0.93 |

**Speedup (wall time, double/single): 2.30x**  
**Speedup (per cycle): 2.29x**

Throughput variability (interval perf, std/median): double 21.8%, single 26.4%.

Cycle count difference: -95 (-0.14 %) single vs double.

## Reference: earlier runs sharing one GPU (tcgm_*1)

| | Double | Single | Double / Single |
|---|---|---|---|
| Wall time [s] | 8760 | 5564 | 1.57 |
| Throughput [Mcell/s] | 31.9 | 50.2 | 0.64 |

Sequential vs shared wall time: double 0.65x, single 0.44x.
