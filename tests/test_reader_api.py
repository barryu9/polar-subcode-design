"""Reader-facing formats, exact enumerators and generic decoding regressions."""
from collections import Counter
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from polar_subcodes.code import PolarSubcode,Design,load_code,load_profile
from polar_subcodes.algebra import polar_transform
from polar_subcodes.io import ROOT,write_json
from polar_subcodes.spectrum.backend import count,words,WorkLimitError
from polar_subcodes.spectrum.decomposition import spectrum
from polar_subcodes.spectrum.quotient import evaluate
from polar_subcodes.construction.source_patterns import PatternSearch,SourcePatternProblem
from polar_subcodes.decoding.scl import decode
from polar_subcodes.decoding.helpers import scl_cpp_available
from polar_subcodes.decoding.simulation import simulate


class ReaderTests(unittest.TestCase):
    def test_lossless_masks_and_canonical_equations(self):
        paths = sorted((ROOT / "configs" / "proposed").glob("*.code.json"))
        self.assertTrue(paths)
        for path in paths:
            code=load_code(path)
            self.assertEqual(code,PolarSubcode(code.n,code.profile,tuple(map(int,code.as_dict()['columns']))))
            self.assertIn('Information positions',code.describe())
            for q in range(min(code.k,10)):
                u=code.input_word(1<<q)
                self.assertTrue(all((u&h).bit_count()%2==0 for h in code.checks))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'support.json'
            write_json(path,dict(N=8,K=4,profile=[1,3,5,7],dynamic_expressions={'2':[1],'4':[1,3]}))
            code=load_code(path)
            self.assertEqual(code.columns[4],10)
            with self.assertRaises(ValueError):
                load_profile(path)
            write_json(path,code.as_dict()|dict(dynamic_expressions={'2':[3]}))
            with self.assertRaises(ValueError):
                load_code(path)

    def test_invalid_profiles_and_supports_are_rejected(self):
        for n,profile,columns in ((6,(1,),(0,)*6),(4,(3,1),(0,)*4),
                                  (4,(1,3),(0,0,8,0)),(4,(1,3),(0,2,0,0)),
                                  (4,(1,3),(0,0,1,0)),(4,(1,3),(0,0,2.0,0))):
            with self.assertRaises(ValueError):
                PolarSubcode(n,profile,columns)

    def test_all_small_supports_match_complete_native_and_quotient_lists(self):
        for n,profile in ((4,(1,3)),(8,(1,3,5,7)),(8,()),(8,tuple(range(8)))):
            edges=[(j,i) for j in range(n) if j not in profile for i in profile if i<j]
            for value in range(1<<len(edges)):
                columns=[0]*n
                for q,(j,i) in enumerate(edges):
                    columns[j]|=((value>>q)&1)<<i
                code=PolarSubcode(n,profile,tuple(columns))
                for cutoff in (0,2,4,n):
                    expected={code.input_word(m):code.encode(m).bit_count() for m in range(1<<code.k)
                              if code.encode(m).bit_count()<=cutoff}
                    actual,meta=words(code,cutoff)
                    self.assertEqual(dict(actual),{u:w for u,w in expected.items() if u})
                    hist=dict(Counter(expected.values()))
                    self.assertEqual(count(code,cutoff)['spectrum'],hist)
                    self.assertEqual(count(code,cutoff,backend='python')['spectrum'],hist)
                    self.assertEqual(spectrum(code,cutoff),hist)
                    for method in ('direct','quotient'):
                        listed,_=evaluate(dict(n=n,checks=code.checks,cutoff=cutoff,kind='directory'),method)
                        self.assertEqual(set(listed),set(expected))
                        self.assertEqual(len(listed),len(set(listed)))

    def test_budget_exhaustion_is_not_a_zero_spectrum(self):
        code=load_code(ROOT/'configs/proposed/n64_k32.code.json')
        with self.assertRaises(WorkLimitError):
            words(code,12,max_work=1)
        with self.assertRaises(WorkLimitError):
            count(code,12,max_work=1)
        with self.assertRaises(RuntimeError):
            evaluate(dict(n=8,checks=(),cutoff=8,kind='directory'),'quotient',1)

    def test_quotient_lifts_match_direct_encoding_for_random_causal_checks(self):
        rng = random.Random(2319)
        for n in (1, 2, 4, 8):
            check_sets = [[], [1 << j for j in range(n)]]
            check_sets.extend([(1 << j) | rng.randrange(1 << j) for j in range(n) if rng.randrange(2)]
                              for _ in range(20))
            for checks in check_sets:
                for cutoff in range(-1, n + 2):
                    case = dict(kind="directory", n=n, checks=checks, cutoff=cutoff)
                    result, _ = evaluate(case, "quotient")
                    expected = {u for u in range(1 << n) if polar_transform(u, n).bit_count() <= cutoff
                                and all((u & row).bit_count() % 2 == 0 for row in checks)}
                    self.assertEqual(set(result), expected)
                    self.assertEqual(len(result), len(expected))

    def test_zero_cutoff_does_not_traverse_the_message_space(self):
        code=PolarSubcode(256,tuple(range(256)),(0,)*256)
        actual,meta=words(code,0,max_work=1)
        self.assertEqual(actual,[])
        self.assertEqual(meta['spectrum'],{0:1})

    def test_decomposed_cutoff_can_exceed_the_constituent_length(self):
        code=PolarSubcode(32,(15,23,27,29,30,31),(0,)*32)
        for cutoff in (17,32):
            expected=dict(Counter(code.encode(m).bit_count() for m in range(1<<code.k)
                                  if code.encode(m).bit_count()<=cutoff))
            self.assertEqual(spectrum(code,cutoff),expected)

    def test_source_pattern_solver_matches_all_small_assignments(self):
        code=PolarSubcode(8,(1,3,5,7),(0,)*8)
        rows=(2,4,6)
        domain=tuple((q,i) for q,j in enumerate(rows) for i in code.profile if i<j)
        mother=PolarSubcode(8,tuple(sorted((*code.profile,*rows))),(0,)*8)
        catalogue,_=words(mother,3)
        problem=SourcePatternProblem(SimpleNamespace(design=Design(8,code.profile),columns=[0]*8,
            rows=rows,domain=domain,words=catalogue))
        solver=PatternSearch(problem,'reverse','product')
        while solver.status=='pending':
            solver.step()
        feasible=[]
        for mask in range(1<<len(domain)):
            columns=[0]*8
            for q,(row,i) in enumerate(domain):
                columns[rows[row]]|=((mask>>q)&1)<<i
            candidate=PolarSubcode(8,code.profile,tuple(columns))
            if all(candidate.encode(m).bit_count()>=4 for m in range(1,1<<candidate.k)):
                feasible.append(mask)
        self.assertEqual(solver.solution is not None,bool(feasible))

    def test_unrestricted_support_size_and_nonzero_messages(self):
        code=PolarSubcode(8,(1,3,5,7),(0,0,2,0,10,0,42,0))
        messages=range(1<<code.k)
        x=np.array([[(code.encode(m)>>i)&1 for i in range(code.n)] for m in messages])
        expected=np.array([[(code.input_word(m)>>i)&1 for i in range(code.n)] for m in messages])
        backends=['numpy']+(['cpp'] if scl_cpp_available() else [])
        for backend in backends:
            np.testing.assert_array_equal(decode(code,20*(1-2*x),16,backend),expected)
        large=load_code(ROOT/'configs/proposed/n64_k32.code.json')
        self.assertGreater(max(map(len,large.supports)),3)
        word=large.encode(1234567)
        llr=np.array([30*(1-2*((word>>i)&1)) for i in range(large.n)])
        expected=np.array([[(large.input_word(1234567)>>i)&1 for i in range(large.n)]])
        for backend in backends:
            np.testing.assert_array_equal(decode(large,llr,8,backend),expected)

    def test_cpp_and_numpy_decisions_agree_on_noisy_frames(self):
        if not scl_cpp_available():
            self.skipTest('Optional C++ SCL extension not built')
        code=load_code(ROOT/'configs/proposed/n64_k32.code.json')
        llr=np.random.default_rng(23).normal(2,3,(20,64))
        for size in (1,4,16):
            np.testing.assert_array_equal(decode(code,llr,size,'numpy'),decode(code,llr,size,'cpp'))

    def test_full_list_matches_exhaustive_ml_on_a_tiny_code(self):
        code=PolarSubcode(4,(1,3),(0,0,2,0))
        llr=np.random.default_rng(144).normal(0,2,(100,4))
        words_array=np.array([[(code.encode(m)>>i)&1 for i in range(4)] for m in range(4)])
        best=np.argmin(llr@words_array.T,axis=1)
        expected=np.array([[(code.input_word(int(m))>>i)&1 for i in range(4)] for m in best])
        np.testing.assert_array_equal(decode(code,llr,4,'numpy'),expected)

    def test_simulation_batch_size_does_not_change_noise_or_counts(self):
        code=PolarSubcode(8,(1,3,5,7),(0,0,2,0,10,0,42,0))
        with tempfile.TemporaryDirectory() as directory:
            first=simulate(code,Path(directory)/'a',0,25,4,1,71,'numpy','random')
            second=simulate(code,Path(directory)/'b',0,25,4,7,71,'numpy','random')
            for key in ('frames','frame_errors','information_bit_errors','FER'):
                self.assertEqual(first[key],second[key])


if __name__=='__main__':
    unittest.main()
