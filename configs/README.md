# Simulation code configurations

English · [简体中文](README.zh-CN.md)

Use `python -m reproduction list` to find a code from Figs. 5–10 by its legend, block length and decoder list size. The [Proposed manifest](proposed/manifest.json) lists the eight Table II cases and the Proposed-NP ablation. The [comparison-code manifest](reference/manifest.json) gives the construction method and inputs for each reference curve.

Each `*.profile.json` contains the information set selected at the start of fixed-profile design. The paired `*.code.json` adds the frozen expressions and the encoding parameters for that curve. The cold construction scripts also reproduce the search that selected the information set. Proposed-NP includes the frozen expressions obtained before prefix protection.

| Field or convention | Interpretation |
|---|---|
| `N`, `profile`, `K` | Block length, distinct zero-based information positions in increasing SC order, and number of information positions. For CRC curves, `payload` gives the user-message length and `K` includes the CRC bits. |
| `dynamic_expressions["j"] = [i, k]` | The frozen bit satisfies $u_j=u_i+u_k$ over the binary field. Each source precedes $j$. An omitted frozen position carries zero. Where available, `columns[j]` encodes the same expression as an integer with bit $i$ set for each source $i$. |
| Polar and PAC encoding | The transform is $x=uF^{\otimes m}$ in natural order. The PAC encoder places payload bits in $v$ and computes $u=v*g$ with taps in increasing delay order. The canonical encoder spans the same PAC code using the information positions of $u$ as its payload coordinates. |
| `crc.polynomial` | Includes its highest-order term. Payload bits enter the register in SC order. The decoder branches on all mother-code information positions and checks the CRC on the final list. If no candidate passes, it reports a frame failure. |

Load a full configuration with `polar_subcodes.codec.load_codec(path)` for encoding or SCL decoding. Use `codec.as_subcode()` for exact weight enumeration of the transmitted code, including the CRC constraints.

SC-ML, RecursCode and Number-polar use the published SC-aimed precoding matrices $Z$. The [portable inputs](../reproduction/reference_inputs) store the rows as hexadecimal integers with position zero in the least significant bit. Each file gives the original filename and SHA-256. Reconstruction row-reduces $Z$ to causal form. The PAC, CRC and eBCH comparison codes are generated from their respective design parameters.
