# Single vs double precision — tcgm run

Grid 4,194,304 cells, t_end = 60 code units, GPU run.

| Quantity | Double | Single | Double / Single |
|---|---|---|---|
| Wall time | 2 hours 26 minutes 0 seconds | 1 hour 32 minutes 44 seconds | 1.57 |
| Wall time [s] | 8760 | 5564 | 1.57 |
| Cycles | 66667 | 66572 | 1.00 |
| Wall time per cycle [ms] | 131.4 | 83.6 | 1.57 |
| Throughput [Mcell-updates/s] | 31.9 | 50.2 | 0.64 |
| Median interval throughput [Mcell/s] | 32.6 | 54.3 | 0.60 |
| Peak GPU memory [MB] | 1195 | 598 | 2.00 |
| Peak host memory [MB] | 549 | 275 | 2.00 |
| Mean VTK write time [s] | 0.102 | 0.048 | 2.14 |
| Mean dump write time [s] | 0.120 | 0.040 | 2.99 |
| Total VTK size [MB] | 5862 | 5862 | 1.00 |
| Total dump size [MB] | 1345 | 673 | 2.00 |
| Median dt | 8.761e-04 | 8.452e-04 | 1.04 |
| Median cooling nsub | 449 | 482 | 0.93 |

**Speedup (wall time, double/single): 1.57x**  
**Speedup (per cycle): 1.57x**

**Caveat:** both runs shared the same GPU until the single run finished (double run at t = 24.7). Double throughput while sharing: 28.2 Mcell/s, alone afterwards: 35.5 Mcell/s. The single run never ran alone, so the speedups above are not a clean benchmark.

Cycle count difference: -95 (-0.14 %) single vs double.
