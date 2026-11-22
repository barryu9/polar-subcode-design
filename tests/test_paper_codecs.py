"""Regression cases where the mother code and transmitted code can differ."""
import unittest
import numpy as np

from polar_subcodes.code import PolarSubcode
from polar_subcodes.codec import PayloadCodec
from polar_subcodes.construction.profiles import make_profile, reliability_order
from polar_subcodes.decoding import full
from polar_subcodes.precoding import append_crc, crc_remainder, pac_code, PAC_TAPS
from reproduction.references import CATALOG, construct, identity, read_json


def llrs(x, n):
    return [(-1 if x >> i & 1 else 1)*(4.1+i/8) for i in range(n)]


class CodecTests(unittest.TestCase):
    def test_pac_information_coordinates(self):
        code = pac_code(16, (3, 4, 5, 8, 9, 12), PAC_TAPS)
        codec = PayloadCodec(code, taps=PAC_TAPS)
        words = [codec.encode(i) for i in range(1 << codec.payload)]
        self.assertEqual(set(words), {code.encode(i) for i in range(1 << code.k)})
        self.assertEqual(codec.decode([llrs(x,16) for x in words], 4, 'numpy'), list(range(64)))

    def test_crc_subcode_and_decoder(self):
        mother = PolarSubcode(8, (2, 3, 4, 6, 7), (0,)*8)
        codec = PayloadCodec(mother, polynomial=0b111)
        words = [codec.encode(i) for i in range(8)]
        reduced = codec.as_subcode()
        self.assertEqual({reduced.encode(i) for i in range(8)}, set(words))
        for i in range(8):
            self.assertEqual(crc_remainder(append_crc(i,3,7),5,7), 0)
        # For L=1 this noiseless *mother* word survives but fails CRC. With
        # a full list the decoder can instead select a CRC-valid competitor.
        corrupted = llrs(mother.encode(1), 8)
        rng = np.random.default_rng(112)
        noisy = rng.normal(0, 1, (9,8))
        backends = ["numpy"] + (["cpp"] if full._SCL_CPP is not None else [])
        for backend in backends:
            self.assertEqual(codec.decode([llrs(x,8) for x in words], 4, backend), list(range(8)))
            self.assertEqual(codec.decode(corrupted, 1, backend), [None])
            self.assertIsNotNone(codec.decode(corrupted, 32, backend)[0])
            # With a full list, ML over transmitted words is an independent oracle.
            best = [max(range(8), key=lambda m: sum(v*(-1 if words[m]>>i&1 else 1)
                        for i,v in enumerate(frame))) for frame in noisy]
            self.assertEqual(codec.decode(noisy, 32, backend), best)

    def test_bec_and_standard_order(self):
        self.assertEqual(reliability_order(4,"bec"), [3,2,1,0])
        self.assertEqual(make_profile(32,0,"5g"), ())
        self.assertEqual(make_profile(32,2,"5g"), (30,31))
        self.assertEqual(make_profile(8,4,"rm_ga"), (3,5,6,7))

    def test_all_published_reference_identities(self):
        rows = read_json(CATALOG)
        self.assertEqual(len(rows),21)
        for row in rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(identity(construct(row)), row["code_hash"])


if __name__ == "__main__":
    unittest.main()
