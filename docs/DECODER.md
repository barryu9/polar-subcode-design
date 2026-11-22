# Decoder and Simulation Notes

The NumPy and C++ implementations use min-sum SC LLR propagation and max-log
path metric increments. On each information bit they retain exactly
`min(L, 2*active_paths)` candidates, breaking ties by candidate bit and parent
path index. Dynamic frozen bits are computed independently on each path.

The public `decode` API disables all-zero-path early termination and returns
the actual selected polar input. It therefore works for nonzero messages.
For the experiment configs, `load_codec` decodes the payload, including
the inverse convolution for PAC. CA-SCL branches on all mother information
positions and selects the minimum-metric valid terminal path. If none passes
the CRC, the frame counts as an error. Bit-error accounting then uses the
minimum-metric candidate. Both backends use the same terminal rule.

## Reader-Release Correction

The research implementation selected candidates by a metric threshold. Ties
could select more candidates than available list slots, leaving an information
decision unupdated for some paths. A noiseless nonzero N64 message with L=8
exposed this during extraction. Both backends in this repository now use stable
top-L selection. The regression includes that exact case, noisy backend agreement,
tiny-code full-list/ML agreement, and supports referencing more than three bits.

The general reader decoder uses the corrected top-L tie rule. The historical
design-frame replay has a separate build of its original tie rule to regenerate
the expression choices used in the plots.

## Statistical Convention

`--snr` denotes information-bit Eb/N0, not Es/N0. With rate `R=K/N`, real BPSK
noise variance is `1/(2 R 10^(EbN0_dB/10))`. Separate seeded streams drive noise
and random messages. The frame count is fixed, so the reported Wilson interval
has its usual fixed-sample interpretation. Zero measured errors give a positive
upper interval endpoint, not a zero-FER guarantee.

The plotted frame/error counts and the paired path-loss measurements are
available in [the r112 data directory](../results/paper_r112/README.md).
