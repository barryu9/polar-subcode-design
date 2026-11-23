# Deterministic Polar Subcode Design via Exact Low-Weight Spectrum Characterization

**English** · [简体中文](README.zh-CN.md)

**Junhua You, Shaohua Wu, Aolin Liu, Yajing Deng, and Qinyu Zhang**

This repository accompanies the paper. To see how the method works, run Examples 3 and 4 on the same length-16 code. Example 3 enumerates the low-weight words directly. Example 4 shows how polar projection recovers the same exclusion conditions. To check a simulation curve, look it up by figure, legend and decoder list size, then inspect its frozen expressions or rebuild it from its construction inputs.

## Quick start

Run the commands from the repository root with Python 3.12. The walkthrough uses Python and NumPy. The larger constructions, native enumerator and accelerated decoder require a C++17 compiler. On Windows, use MinGW for enumeration and Microsoft C++ Build Tools for the SCL extension. On Linux use GCC, and on macOS use Clang.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m paper_examples 3
python -m paper_examples 4
python -m reproduction list
python -m reproduction check
python -m reproduction.table_ii_values
```

## Read alongside the paper

| Paper | Entry point | What to inspect |
|---|---|---|
| Algorithms 1–4 | [Algorithm guide](algorithms/README.md) | Recursive enumeration, source-wise completion, fixed-profile search and information-set exchanges. Follow an unresolved suffix from Algorithm 3 into Algorithm 4. |
| Examples 1–6 | [Example guide](paper_examples/README.md) · `python -m paper_examples 5` | Reproduce the length-16 hand calculations, the projection ranks and the recovery of the four feasible coefficient configurations. |
| Example 7 | `python -m paper_examples 7` | Compare the source-wise counting bound with the exact count for the length-32 example. |
| Examples 8–9 | `python -m paper_examples 8` · `python -m paper_examples 9` | Follow fixed-profile construction and information-set exchanges at length 256. Example 8 uses a different information set from Fig. 8. Example 9 constructs the Proposed code in Fig. 7. The example guide gives the commands to rebuild both and verify their spectra. |
| Theorems 1–3, Propositions 1–3 and Corollaries 1–2 | [Theory checks](theory/README.md) · `python -m theory.check_results` | Follow eight proof arguments through exact calculations on finite examples. |
| Section VI, Figs. 5–10 and Tables II–III | [Configurations](configs/README.md) · [Measurements](results/README.md) | Match each curve to its information set, frozen expressions, decoder list size and measured frame error rate (FER). |

```bash
python -m algorithms.algorithm1_directory
python -m algorithms.algorithm2_outer
python -m algorithms.algorithm3_fixed_profile
python -m algorithms.algorithm4_profiles
python -m paper_examples all
python -m theory.check_results theorem2 proposition3
```

## Rebuild the simulation codes

The last command below constructs all **30 codes** in Figs. 5–10 in a new output directory. To reproduce just one curve, select its case ID with `--cases`. Use `python -m reproduction list` to see the available IDs.

```bash
python -m pip install -r reproduction/table_ii/requirements.txt
python -m reproduction construct --cases n256_k64 --output runs/fig7-proposed
python -m reproduction construct --cases n64_k32_crc4_ga n64_k32_sc_ml --output runs/fig5-refs
python -m reproduction construct --output runs/all-curves
```

**Nine Proposed codes.** Each run starts with an empty candidate archive, follows the recorded search and refinement decisions, then checks the resulting information positions and frozen expressions against those used for the figure. Table II times eight construction runs. The ninth code, Proposed-NP in Fig. 10, is taken from the main run just before prefix protection. The verified outputs appear under `proposed/selected/`. When a search has a wall-time limit, the machine can affect which branches it reaches. The final identity check reports a mismatch if the recorded code was not reached.

**Twenty-one comparison codes.** SC-ML, RecursCode and Number-polar are reconstructed from their published precoding matrices. PAC, CRC and eBCH codes are constructed from their design parameters. These outputs appear under `reference/`.

**Configurations and timing.** Each `*.profile.json` specifies the information set. The paired `*.code.json` gives the frozen expressions and any CRC or PAC parameters. All indices start at zero. Run `python -m reproduction check` to verify the distributed configurations. [Configuration format and bit order](configs/README.md)

On the original experiment machine, the eight runs took about 100 minutes in sequence, counting the shared main-code stage once. In Table II, **w/o refinement** measures the time to obtain a code that meets the minimum-distance target while taking channel reliability into account. **w/ refinement** also includes spectral refinement and prefix protection. The latter uses decoding on design frames. FER is evaluated on noise samples separate from the design frames. [Measurement tables](results/README.md)

## Library

| Operation | Implementation |
|---|---|
| GA, RM–GA, BEC and 5G NR information sets | [profiles.py](polar_subcodes/construction/profiles.py) |
| Canonical polar subcodes, binary linear algebra and the polar transform | [code.py](polar_subcodes/code.py), [algebra.py](polar_subcodes/algebra.py) |
| PAC and CRC encoding, precoding matrix conversion and eBCH construction | [precoding.py](polar_subcodes/precoding.py), [ebch.py](polar_subcodes/construction/ebch.py) |
| Payload encoding, SC, SCL and CRC-aided selection from the final list | [codec.py](polar_subcodes/codec.py), [NumPy SCL](polar_subcodes/decoding/full.py), [C++ SCL](cpp/scl_step_decoder_cpp.cpp) |
| Exact low-weight spectra and complete codeword lists | [Spectrum routines](polar_subcodes/spectrum/), [Algorithm 1](algorithms/algorithm1_directory.py) |

For the paper's spectrum counts, this repository bundles the Python and C++ code from version 0.1.0 of our spectrum enumerator. The bundled C++ kernel has configurable capacity for the length-1,024 codes. The standalone project also provides a MATLAB implementation and examples for puncturing and shortening. [polar-low-weight-spectrum](https://github.com/barryu9/polar-low-weight-spectrum)

The following example encodes a nonzero payload and decodes its noiseless log-likelihood ratios with the NumPy SCL decoder.

```python
from polar_subcodes.codec import load_codec

code = load_codec("configs/reference/n64_k32_crc4_ga.code.json")
x = code.encode(37)  # 32 payload bits, including leading zeros
llr = [12.0 * (1 - 2*((x >> i) & 1)) for i in range(code.mother.n)]
assert code.decode(llr, list_size=32, backend="numpy") == [37]
```

```bash
python tools/build_native.py --scl
python main.py inspect configs/proposed/n64_k32.code.json
python main.py enumerate --support configs/proposed/n64_k32.code.json --cutoff 12 --method native --output runs/fig5-spectrum
python main.py enumerate --support configs/proposed/n64_k32.code.json --cutoff 12 --method words --output runs/fig5-words
python main.py simulate --support configs/reference/n64_k32_crc4_ga.code.json --snr 3 --list-size 32 --frames 100 --messages random --backend cpp --output runs/crc-smoke
```

For the Fig. 5 Proposed code, the first positive spectrum coefficient is $A_{12}=2048$. The two enumeration commands compute this count and list the corresponding words. The simulation command uses fresh noise samples. Both decoder backends accept nonzero payloads and use the payload rate to set $E_b/N_0$. Select `--backend numpy` for Python or `--list-size 1` for SC. [Decoder and simulation conventions](docs/DECODER.md) · [Validation checks](docs/VALIDATION.md)

## References

The following sources describe the techniques used in the library and the construction of the comparison codes.

- E. Arıkan, “Channel Polarization,” IEEE Trans. Inf. Theory, 2009. [Polar transform and channel recursion](https://doi.org/10.1109/TIT.2009.2021379)
- P. Trifonov and V. Miloslavskaya, “Polar Codes With Dynamic Frozen Symbols and Their Decoding by Directed Search,” ITW 2013, and “Polar Subcodes,” IEEE JSAC, 2016. [Dynamic frozen symbols](https://doi.org/10.1109/ITW.2013.6691213) · [eBCH subcodes](https://doi.org/10.1109/JSAC.2015.2504269)
- P. Trifonov, “Efficient Design and Decoding of Polar Codes,” IEEE Trans. Commun., 2012. S.-Y. Chung, T. Richardson and R. Urbanke, IEEE Trans. Inf. Theory, 2001, for the fitted GA check-node approximation. [Polar GA](https://doi.org/10.1109/TCOMM.2012.081512.110872) · [GA approximation](https://doi.org/10.1109/18.910580)
- I. Tal and A. Vardy, “List Decoding of Polar Codes,” IEEE Trans. Inf. Theory, 2015. E. Arıkan, “From Sequential Decoding to Channel Polarization and Back Again,” IEEE ITS Newsletter, 2019, for PAC. ETSI TS 138 212 V19.4.0, §§5.1, 5.3.1.2, for CRC6, CRC11 and the NR mother-code reliability order. [List decoding](https://doi.org/10.1109/TIT.2015.2410251) · [5G standard](https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/19.04.00_60/ts_138212v190400p.pdf)
- V. Miloslavskaya and coauthors, IEEE Trans. Commun., 2020, 2021 and 2024, for the published precoding matrices. [SC-ML](https://doi.org/10.1109/TCOMM.2020.3014946) · [RecursCode](https://doi.org/10.1109/TCOMM.2021.3111625) · [Number-polar](https://doi.org/10.1109/TCOMM.2023.3331532)

See the [notices](NOTICE.md) for code and data attribution.

## AI Statement

OpenAI's GPT-6 organized this repository from the authors' local experimental code, construction records and manuscript to help readers and prospective reviewers understand and reproduce the work and build on it in further research.
