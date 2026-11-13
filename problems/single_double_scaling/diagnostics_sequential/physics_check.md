# Physics check: single vs double precision

Relative difference (single - double) / double of domain totals over 61 snapshots.

| Quantity | max abs rel. diff | rel. diff at t=60 | median abs rel. diff |
|---|---|---|---|
| total mass | 9.96e-04 (t=59) | -9.63e-04 | 3.56e-04 |
| total energy | 6.78e-04 (t=41) | -6.28e-04 | 3.04e-04 |
| wind (tracer) mass | 3.53e-04 (t=1) | -5.13e-05 | 5.34e-05 |
| cold mass (T<2e+04 K) | 7.12e-03 (t=28) | -4.21e-03 | 1.77e-03 |
| hot mass (T>2e+06 K) | 7.29e-02 (t=60) | -7.29e-02 | 1.47e-03 |

Wind gas first reaches a boundary at t = inf.

## Tracer mass vs injected mass (before wind leaves the box)

| t | injected [Msun] | double [Msun] | single [Msun] | double/inj - 1 | single/inj - 1 |
|---|---|---|---|---|---|
| 1 | 9.5887e+04 | 8.9715e+05 | 8.9683e+05 | +8.36e+00 | +8.35e+00 |
| 5 | 4.4313e+05 | 4.9040e+06 | 4.9037e+06 | +1.01e+01 | +1.01e+01 |
| 10 | 8.0591e+05 | 8.1583e+06 | 8.1580e+06 | +9.12e+00 | +9.12e+00 |
| 15 | 1.1029e+06 | 9.6183e+06 | 9.6177e+06 | +7.72e+00 | +7.72e+00 |
| 20 | 1.3461e+06 | 1.0304e+07 | 1.0303e+07 | +6.65e+00 | +6.65e+00 |
| 25 | 1.5452e+06 | 1.0666e+07 | 1.0666e+07 | +5.90e+00 | +5.90e+00 |
| 30 | 1.7083e+06 | 1.0876e+07 | 1.0876e+07 | +5.37e+00 | +5.37e+00 |
| 35 | 1.8417e+06 | 1.1020e+07 | 1.1020e+07 | +4.98e+00 | +4.98e+00 |
| 40 | 1.9510e+06 | 1.1130e+07 | 1.1130e+07 | +4.70e+00 | +4.70e+00 |
| 45 | 2.0404e+06 | 1.1220e+07 | 1.1219e+07 | +4.50e+00 | +4.50e+00 |
| 50 | 2.1137e+06 | 1.1293e+07 | 1.1293e+07 | +4.34e+00 | +4.34e+00 |
| 55 | 2.1736e+06 | 1.1353e+07 | 1.1353e+07 | +4.22e+00 | +4.22e+00 |
| 60 | 2.2227e+06 | 1.1403e+07 | 1.1402e+07 | +4.13e+00 | +4.13e+00 |

## Totals at t = 60

| Quantity | double | single |
|---|---|---|
| total mass | 2.897255e+03 | 2.894464e+03 |
| total energy | 1.315990e+03 | 1.315164e+03 |
| wind (tracer) mass | 4.615161e-01 | 4.614925e-01 |
| cold mass (T<2e+04 K) | 2.749956e+01 | 2.738387e+01 |
| hot mass (T>2e+06 K) | 1.892055e-03 | 1.754208e-03 |
