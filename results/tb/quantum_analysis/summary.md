# Quantum circuit analysis

## Expressibility and entangling capability (4 qubits)

| entanglement   |   depth |   expressibility_kl |   entangling_capability |
|:---------------|--------:|--------------------:|------------------------:|
| strong         |       1 |              0.3082 |                  0.7919 |
| strong         |       2 |              0.0083 |                  0.8403 |
| strong         |       3 |              0.0047 |                  0.8158 |
| strong         |       4 |              0.0035 |                  0.8242 |
| basic          |       1 |              0.6793 |                  0.8441 |
| basic          |       2 |              0.0202 |                  0.8743 |
| basic          |       3 |              0.0047 |                  0.8745 |
| basic          |       4 |              0.0054 |                  0.8746 |
| none           |       1 |              0.2955 |                  0.0000 |
| none           |       2 |              0.1910 |                  0.0000 |
| none           |       3 |              0.2056 |                  0.0000 |
| none           |       4 |              0.2057 |                  0.0000 |

## Barren plateaus (StronglyEntanglingLayers, random init)

|   n_qubits |   ('global', 2) |   ('global', 10) |   ('local', 2) |   ('local', 10) |
|-----------:|----------------:|-----------------:|---------------:|----------------:|
|          2 |       3.338e-02 |        1.220e-01 |      2.330e-01 |       1.501e-01 |
|          4 |       5.416e-02 |        3.201e-02 |      6.615e-02 |       3.112e-02 |
|          6 |       6.043e-03 |        8.953e-03 |      2.093e-03 |       7.598e-03 |
|          8 |       3.646e-03 |        2.106e-03 |      6.795e-03 |       2.029e-03 |
|         10 |       6.357e-04 |        5.015e-04 |      2.195e-04 |       4.221e-04 |

## Trained C->Q model on realistic hardware (test set)

| setting      |      shots |   noise_p |   accuracy |   recall |     f1 |   roc_auc |
|:-------------|-----------:|----------:|-----------:|---------:|-------:|----------:|
| exact        |   nan      |    0.0000 |     0.9841 |   0.9333 | 0.9515 |    0.9987 |
| shots        |    10.0000 |    0.0000 |     0.9825 |   0.9333 | 0.9469 |    0.9979 |
| shots        |    50.0000 |    0.0000 |     0.9857 |   0.9333 | 0.9561 |    0.9980 |
| shots        |   100.0000 |    0.0000 |     0.9857 |   0.9333 | 0.9561 |    0.9985 |
| shots        |   500.0000 |    0.0000 |     0.9825 |   0.9238 | 0.9463 |    0.9987 |
| shots        |  1000.0000 |    0.0000 |     0.9841 |   0.9333 | 0.9515 |    0.9986 |
| shots        | 10000.0000 |    0.0000 |     0.9825 |   0.9238 | 0.9463 |    0.9986 |
| depolarizing |   nan      |    0.0010 |     0.9841 |   0.9333 | 0.9515 |    0.9987 |
| depolarizing |   nan      |    0.0050 |     0.9841 |   0.9333 | 0.9515 |    0.9987 |
| depolarizing |   nan      |    0.0100 |     0.9841 |   0.9333 | 0.9515 |    0.9987 |
| depolarizing |   nan      |    0.0200 |     0.9841 |   0.9333 | 0.9515 |    0.9987 |
| depolarizing |   nan      |    0.0500 |     0.9841 |   0.9238 | 0.9510 |    0.9987 |
| depolarizing |   nan      |    0.1000 |     0.9841 |   0.9238 | 0.9510 |    0.9987 |
| depolarizing |   nan      |    0.2000 |     0.9857 |   0.9238 | 0.9557 |    0.9987 |

## Gradient check

Max |backprop - parameter-shift| over all circuit weights: 1.11e-15 (parameter-shift is what real hardware would use).
