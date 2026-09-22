# Data and Transform Conventions

- All positions are zero based, sorted in increasing SC order.
- `N` is a power of two. `K` is the number of information positions.
- `x = u G_N`, `G_N = F^{tensor log2(N)}`, `F = [[1,0],[1,1]]`.
- There is no extra input or output bit-reversal permutation.
- Integer bit `j` stores vector coordinate `j` (least-significant bit first).
- At an information position, `columns[j]` is zero.
- At a frozen position, `u[j]` is the XOR of information bits selected by
  `columns[j]`. Every selected position must be less than `j`.
- `d_P = min(2^popcount(i) for i in profile)` is the zero-support distance.

Minimal profile:

```json
{"N": 8, "K": 4, "profile": [1, 3, 5, 7]}
```

Equivalent support encodings:

```json
{"N": 8, "K": 4, "profile": [1, 3, 5, 7],
 "columns": ["0", "0", "2", "0", "10", "0", "42", "0"]}
```

```json
{"N": 8, "K": 4, "profile": [1, 3, 5, 7],
 "dynamic_expressions": {"2": [1], "4": [1, 3], "6": [1, 3, 5]}}
```

These specify `u[2]=u[1]`, `u[4]=u[1] XOR u[3]` and
`u[6]=u[1] XOR u[3] XOR u[5]`. Unspecified frozen expressions are zero.
Use decimal strings for masks to avoid loss of precision in JavaScript/JSON tools.

```python
from polar_subcodes.code import load_code, PolarSubcode
from polar_subcodes.decoding.scl import decode
import numpy as np

code = load_code('results/n64_k32/support.json')
message = 123
x = code.encode(message)
llr = np.array([20*(1-2*((x >> j) & 1)) for j in range(code.n)])
u_hat = decode(code, llr, list_size=8)
information_hat = u_hat[:, code.profile]
print(code.describe())
```

To search a custom information set, pass a profile-only JSON to `construct`.
The constructor refuses a support file as its profile input. To simulate a
custom code, pass its support JSON directly to `simulate`. Existing result
directories are never silently overwritten by experiment commands.
