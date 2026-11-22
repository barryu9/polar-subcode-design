"""Payload encoders and SC/SCL decoding for the published experiment configs.

The mother code fixes a polar input's frozen positions. PAC specifies how a
payload maps to the free positions. A CRC additionally restricts the payload
subspace but leaves its check bits free during SCL branching.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .algebra import polar_transform
from .code import PolarSubcode, integer, load_code
from .decoding import full
from .decoding.scl import decode, parameters
from .precoding import append_crc, crc_remainder, from_input_basis, pac_code, pac_input, unpack_pac


@dataclass(frozen=True)
class PayloadCodec:
    mother: PolarSubcode
    polynomial: int | None = None
    taps: tuple = ()

    def __post_init__(self):
        if self.polynomial is not None:
            crc_remainder(0, 0, self.polynomial)
            if self.payload < 1:
                raise ValueError("CRC leaves no payload positions")
        if self.taps and pac_code(self.mother.n, self.mother.profile, self.taps) != self.mother:
            raise ValueError("Frozen expressions disagree with the declared PAC polynomial")

    @property
    def payload(self):
        width = self.polynomial.bit_length()-1 if self.polynomial is not None else 0
        return self.mother.k - width

    def input_word(self, message):
        if self.polynomial is not None:
            message = append_crc(message, self.payload, self.polynomial)
        if self.taps:
            return pac_input(self.mother.n, self.mother.profile, message, self.taps)
        return self.mother.input_word(message)

    def encode(self, message):
        return polar_transform(self.input_word(message), self.mother.n)

    def as_subcode(self):
        """The transmitted linear code, including CRC, for exact weight enumeration."""
        if self.polynomial is None:
            return self.mother
        return from_input_basis(self.mother.n, [self.input_word(1 << i) for i in range(self.payload)])

    def _information_word(self, row):
        if self.taps:
            u = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
            return unpack_pac(u, self.mother.profile, self.taps)
        return sum(int(row[i]) << q for q, i in enumerate(self.mother.profile))

    def decode(self, llr, list_size=32, backend="auto"):
        """Return a payload or None (no surviving CRC-valid path) for each frame."""
        payloads, valid = self.decode_with_status(llr, list_size, backend)
        return [word if ok else None for word, ok in zip(payloads, valid)]

    def decode_with_status(self, llr, list_size=32, backend="auto"):
        """Return candidate payloads and a Boolean acceptance flag per frame. L=1 is SC.

        For CA-SCL, pick the least-metric surviving CRC-valid path. If no path
        passes the check, flag a frame error and use the least-metric candidate
        for bit-error accounting, following the simulation convention.
        """
        if self.polynomial is None:
            words = [self._information_word(row) for row in decode(self.mother, llr, list_size, backend)]
            return words, np.ones(len(words), dtype=bool)
        data = np.atleast_2d(np.asarray(llr, dtype=np.float64))
        n = self.mother.n
        if data.ndim != 2 or data.shape[1] != n or not np.all(np.isfinite(data)):
            raise ValueError("LLRs must be finite and have shape [frames,N]")
        if type(list_size) is not int or list_size < 1:
            raise ValueError("List size must be a positive integer")
        params = parameters(n)
        info = set(self.mother.profile)
        actions = [0 if j in info else 2 if c else 1 for j, c in enumerate(self.mother.columns)]
        ops, expressions = full.pack_construction_expressions(actions, self.mother.supports, None)
        native = full.full_scl_backend_name(backend) == "cpp"
        if native and full._SCL_CPP is None:
            raise ImportError("Build C++ SCL with python tools/build_native.py --scl")
        result = []
        valid = []
        for frame in data:
            batch = np.ascontiguousarray(frame[None, :])
            if native:
                paths, metrics, active = full._SCL_CPP.decode_prefix_list_pxb(
                    n, params.lambda_offset, params.llr_layer_vec, params.bit_layer_vec,
                    ops, expressions, batch, list_size, n)
            else:
                batch_paths, batch_metrics, batch_active = full._decode_numpy(
                    params, ops, expressions, batch, list_size, False, 0.0, False, return_list=True)
                paths, metrics, active = batch_paths[0], batch_metrics[0], batch_active[0]
            alive = np.flatnonzero(active)
            order = alive[np.argsort(metrics[alive], kind="stable")]
            messages = (self._information_word(paths[i]) for i in order)
            best = self._information_word(paths[order[0]])
            passed = False
            for word in messages:
                if crc_remainder(word, self.mother.k, self.polynomial) == 0:
                    best = word
                    passed = True
                    break
            result.append(best & ((1 << self.payload)-1))
            valid.append(passed)
        return result, np.asarray(valid, dtype=bool)

    def as_dict(self):
        data = self.mother.as_dict()
        data["payload"] = self.payload
        if self.polynomial is not None:
            data["crc"] = {"width": self.polynomial.bit_length()-1, "polynomial": hex(self.polynomial)}
        if self.taps:
            data["pac_generator"] = list(self.taps)
        return data


def load_codec(path):
    """Read a profile/final canonical JSON, including any outer CRC or PAC mapping."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    crc = data.get("crc")
    polynomial = integer(crc["polynomial"]) if crc else None
    if crc and crc["width"] != polynomial.bit_length()-1:
        raise ValueError("CRC width and polynomial disagree")
    if data.get("systematic") or data.get("bit_reversal"):
        raise ValueError("This codec uses the paper's nonsystematic, natural-order transform")
    codec = PayloadCodec(load_code(path), polynomial, tuple(data.get("pac_generator", ())))
    if data.get("payload", codec.payload) != codec.payload:
        raise ValueError("Declared payload length disagrees with the profile and CRC")
    return codec
