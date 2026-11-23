# Reproduce a code from Figs. 5–10

English · [简体中文](README.zh-CN.md)

Run these commands from the repository root in a Python 3.12 environment. Start with the figure and legend in the paper. The `list` command gives the corresponding case ID and SCL list size `L`. Use the case ID to follow the same code through its construction inputs, final frozen expressions and FER measurements.

```sh
python -m pip install -r requirements.txt
python -m reproduction list
python -m reproduction check
python -m reproduction.table_ii_values
```

`check` reads all 30 supplied profile/code pairs and the saved FER points. It checks the code identity, payload size, decoder list size and the ratio of errors to frames. The last command reproduces three columns of Table II from the information sets alone. For example, for Fig. 7 it gives `d_P=32` for **Proposed** and the GA profile statistics 0.113066 (**Proposed**) and 0.194171 (**RecursCode**) at 2.25 dB. The minimum distance 48 in that row refers to the codes after precoding. [Table II and FER measurements](../results/README.md)

## Rebuild a comparison code

```sh
python -m reproduction construct --cases n64_k32_sc_ml --output runs/fig5-sc-ml
```

For **SC-ML** in Fig. 5, [references.py](references.py) reads the published precoding matrix $Z$, stored as one hexadecimal integer per row. Bit $j$ gives the entry at polar-input position $j$. Row reduction identifies the information set and writes each frozen bit as a sum of earlier information bits, as in Section II. The `reference/` output contains `n64_k32_sc_ml.profile.json` with the information set and `n64_k32_sc_ml.code.json` with its dynamic frozen expressions. The [matrix input](reference_inputs/Spec_n64_k32_d12_Z_2G.json) records the source filename and SHA-256 for comparison with the original specification.

**RecursCode** and **Number-polar** use the same matrix route. For **PAC**, **CRC** and **eBCH polar subcode** curves, the driver uses the parameters in the [reference manifest](../configs/reference/manifest.json). For CRC-aided SCL decoding, the profile includes the CRC positions and the final configuration specifies the polynomial.

## Rebuild a Proposed code

```sh
python -m pip install -r reproduction/table_ii/requirements.txt
python -m reproduction construct --cases n256_k64 --output runs/fig7-proposed
```

This case follows Example 9 to the **Proposed** curve in Fig. 7. The driver computes bit-channel reliabilities, searches information sets and constructs the dynamic frozen expressions for the distance target 48. After construction, [proposed.py](proposed.py) checks the information set and every frozen expression against the Fig. 7 code. Open `proposed/selected/n256_k64.code.json` under the output directory to inspect the result. Use a fresh output directory for each construction. Omitting `--cases` reconstructs all 30 codes. The [Table II guide](table_ii/README.md) gives compiler requirements and links to each case driver.

The generated `proposed/table_ii_times.csv` gives the two cumulative times. **w/o refinement** covers the reliability-guided construction that already yields a distance-qualified code and includes separately timed assembly and verification. **w/ refinement** adds spectral refinement, decoding-prefix protection and their joint feedback where applicable. At length 1,024, each **Proposed-LM** or **Proposed-D32** row includes the shared **Proposed** run once. **Proposed-NP** comes from that run before decoding-prefix protection. Section VI evaluates the selected codes on independent noise samples. The saved curve points can be read directly from [fer.csv](../results/fer.csv).
