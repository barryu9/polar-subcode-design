"""Exhaustive low-weight rejection and outer-completion mechanisms.

No random sampling: compare complete message enumeration with the manuscript's
mother-pattern and quotient-event representations at every support assignment.
Run from any directory; generated figures are native pgfplots, not screenshots.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from polar_subcodes.theory.interfaces import (
    count_interfaces, normalize_event, polar_outer_events,
)
from polar_subcodes.algebra import polar_transform

ROOT = Path(__file__).resolve().parents[1]


def scalar_words(profile, columns):
    words = []
    for message in range(1 << len(profile)):
        u = sum(((message >> q) & 1) << i for q, i in enumerate(profile))
        for j, support in enumerate(columns):
            u |= ((u & support).bit_count() & 1) << j
        words.append((u, polar_transform(u, len(columns)).bit_count()))
    return words


RIGHT = (3, 5, 9, 10, 11, 12, 13, 14, 15)


def outer_directory(left, right, h=16):
    """Enumerate the full mother directly, without quotient weight formulas.

    Every physical right word occurs once for each left message. Its input
    syndrome identifies Tm. The resulting integer histograms recover every
    candidate's entire spectrum, without listing it repeatedly at each T.
    """
    frozen = [j for j in range(h) if j not in right]
    dimension = len(left)*len(frozen)
    parameters = np.arange(1 << dimension, dtype=np.uint32)
    profile = tuple(left)+tuple(h+i for i in right)
    physical = np.arange(1 << h, dtype=np.uint32)
    wt = np.array([v.bit_count() for v in range(1 << h)], dtype=np.uint8)
    inputs = np.array([polar_transform(int(v), h) for v in physical], dtype=np.uint32)
    syndromes = np.zeros(len(physical), dtype=np.uint32)
    for q, j in enumerate(frozen):
        syndromes |= ((inputs >> j) & 1) << q
    tables, coset_minimum, t_messages = {}, {}, {}
    for message in range(1, 1 << len(left)):
        u = sum(((message >> q) & 1) << i for q, i in enumerate(left))
        a = polar_transform(u, h)
        weights = wt[physical ^ a]+wt[physical]
        counts = np.zeros((1 << len(frozen), 2*h+1), dtype=np.uint32)
        np.add.at(counts, (syndromes, weights), 1)
        assert np.all(counts.sum(axis=1) == 1 << len(right))
        tables[u] = counts
        values = np.zeros(len(parameters), dtype=np.uint32)
        for t in range(len(left)):
            if message >> t & 1:
                values ^= (parameters >> (len(frozen)*t)) & ((1 << len(frozen))-1)
        t_messages[u] = values
        coset_minimum[u] = (counts != 0).argmax(axis=1)[values]
    right_spectrum = Counter(w for _, w in scalar_words(right, [0]*h))
    pure_minimum = 2*min(w for w in right_spectrum if w)
    minimum = np.full(len(parameters), pure_minimum, dtype=np.uint8)
    for values in coset_minimum.values():
        minimum = np.minimum(minimum, values)
    # Full scalar encodings check the packing and counts at deterministic samples.
    from polar_subcodes.theory.interfaces import lift_outer_parameter
    for parameter in (0, 1, len(parameters)//3, len(parameters)-1):
        columns = lift_outer_parameter(profile, [0]*(2*h), parameter)
        actual = Counter(w for _, w in scalar_words(profile, columns))
        predicted = Counter({2*w: count for w, count in right_spectrum.items()})
        for u, counts in tables.items():
            predicted.update({w: int(count) for w, count in enumerate(counts[t_messages[u][parameter]]) if count})
        assert actual == predicted
    return dict(n=2*h, left=left, right=right, parameters=parameters,
                dimension=dimension, profile=profile, tables=tables,
                coset_minimum=coset_minimum, t_messages=t_messages, minimum=minimum,
                right_distance=pure_minimum//2)


def replacement(directory):
    total = np.zeros(len(directory['parameters']), dtype=np.uint32)
    old = total.copy()
    baseline = 0
    for u, table in directory['tables'].items():
        values = directory['t_messages'][u]
        total += table[values, 4]
        old += (values == 0)*table[0, 4]
        baseline += int(table[0, 4])
    histogram = Counter(zip(old.tolist(), total.tolist()))
    return dict(n=32, profile=directory['profile'], left_profile=directory['left'],
                right_profile=directory['right'], internal_support="zero",
                dimension=directory['dimension'], assignments=len(total), baseline_a4=baseline,
                histogram=[dict(old_survivors=a, total_a4=b, supports=count)
                           for (a, b), count in sorted(histogram.items())],
                false_eliminations=sum(count for (a, b), count in histogram.items() if a == 0 and b > 0),
                true_eliminations=histogram[0, 0],
                validation="complete mother input counts at every T; independent scalar spectra at four fixed T")


def event_values(parameters, event):
    values = np.zeros(len(parameters), dtype=np.uint32)
    for k, row in enumerate(event['rows']):
        parity = np.zeros(len(parameters), dtype=np.uint32)
        while row:
            bit = row & -row
            parity ^= (parameters >> (bit.bit_length()-1)) & 1
            row ^= bit
        values |= parity << k
    return values ^ event.get('offset', 0)


def verified_events(directory, target):
    left, parameters, dimension, profile = (directory[k] for k in ('left', 'parameters', 'dimension', 'profile'))
    critical = [u for u, w in scalar_words(left, [0]*(directory['n']//2)) if 0 < w < target]
    events, actual_dimension = polar_outer_events(profile, [0]*directory['n'], critical, target-1)
    assert actual_dimension == dimension
    violations = []
    for event in events:
        mask = np.isin(event_values(parameters, event), list(event['forbidden']))
        assert np.array_equal(mask, directory['coset_minimum'][event['left_input']] < target)
        violations.append(mask)
    return events, violations


def overlap():
    left, right = (3, 5, 7), RIGHT
    directory = outer_directory(left, right)
    parameters, dimension = directory['parameters'], directory['dimension']
    events, masks = verified_events(directory, 8)
    violations = np.vstack(masks)
    full = int(np.count_nonzero(~violations.any(axis=0)))
    assert full == int(np.count_nonzero(directory['minimum'] >= 8))
    cases = []
    for omit in range(len(events)):
        indices = [j for j in range(len(events)) if j != omit]
        count = int(np.count_nonzero(~violations[indices].any(axis=0)))
        joint = count_interfaces([events[j] for j in indices], dimension)
        assert count == joint['successful_parameters']
        cases.append(dict(omitted=omit+1, feasible=count))
    assert full == count_interfaces(events, dimension)['successful_parameters']
    return dict(n=32, left_profile=left, right_profile=right,
                internal_support="zero", target=8, dimension=dimension,
                assignments=len(parameters), critical_left_inputs=[e['left_input'] for e in events],
                leave_one_out=cases, all_conditions=full,
                validation="every quotient event equals its directly enumerated coset; joint counts agree")


def as_fraction(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def exact_mean(events, dimension, fixed_mask=0, prefix=0):
    """Exact affine-image mean, retaining free parameter coordinates in place."""
    mean = Fraction(0)
    for event in events:
        offset = event.get('offset', 0)
        for k, row in enumerate(event['rows']):
            offset ^= ((row & prefix).bit_count() & 1) << k
        restricted = normalize_event(dict(rows=tuple(r & ~fixed_mask for r in event['rows']),
                                          forbidden=event['forbidden'], offset=offset), dimension)
        mean += Fraction(len(restricted['forbidden']), 1 << len(restricted['rows']))
    return mean


def completion(directory):
    left, right, parameters, dimension, minimum, right_distance = (
        directory[k] for k in ('left', 'right', 'parameters', 'dimension', 'minimum', 'right_distance'))
    targets, trace_events, trace_violations = [], None, None
    for target in range(4, 10):
        events, masks = verified_events(directory, target)
        # A constant event includes pure-right infeasibility above its distance cap.
        if target > 2*right_distance:
            events.append(dict(rows=(), forbidden={0}))
            masks.append(np.ones(len(parameters), dtype=bool))
        violations = np.vstack(masks).sum(axis=0) if masks else np.zeros(len(parameters), dtype=int)
        good = int(np.count_nonzero(violations == 0))
        assert good == int(np.count_nonzero(minimum >= target))
        phi = exact_mean(events, dimension)
        assert phi == Fraction(int(violations.sum()), len(parameters))
        bound = max(Fraction(0), 1-phi)
        assert bound <= Fraction(good, len(parameters))
        targets.append(dict(target=target, feasible=good, phi=as_fraction(phi),
                            lower_bound=as_fraction(bound)))
        if target == 6:
            trace_events, trace_violations = events, violations
    prefix = fixed_mask = 0
    trace = []
    for depth in range(dimension+1):
        mask = (parameters & fixed_mask) == prefix
        phi = exact_mean(trace_events, dimension, fixed_mask, prefix)
        assert phi == Fraction(int(trace_violations[mask].sum()), int(mask.sum()))
        trace.append(dict(fixed_bits=depth, prefix=prefix, phi=as_fraction(phi)))
        if depth < dimension:
            next_mask = fixed_mask | (1 << depth)
            means = [exact_mean(trace_events, dimension, next_mask, prefix | (b << depth))
                     for b in (0, 1)]
            bit = int(means[1] < means[0])
            assert means[bit] <= phi
            prefix |= bit << depth
            fixed_mask = next_mask
    assert trace_violations[prefix] == 0 and minimum[prefix] >= 6
    return dict(n=32, left_profile=left, right_profile=right, internal_support="zero",
                dimension=dimension, assignments=len(parameters), right_distance=right_distance,
                targets=targets, trace_target=6, trace=trace, chosen_parameter=prefix,
                chosen_distance=int(minimum[prefix]),
                parameter_order="increasing frozen destination within increasing source; bit 0 on ties",
                validation="all actual codeword distances equal complete quotient-event tests; affine means exact")


def coordinates(pairs):
    return ' '.join(f'({x},{y:.9g})' for x, y in pairs)


def fraction_value(value):
    return value['numerator']/value['denominator']


def figures(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    replacement_data = report['replacement']['histogram']
    points = '\n'.join(f"{r['old_survivors']} {r['total_a4']} {r['supports']}"
                       for r in replacement_data)
    bars = coordinates([(c['omitted'], c['feasible']) for c in report['overlap']['leave_one_out']]+[(7, 0)])
    directory.joinpath('toy_elimination.tex').write_text(r"""% Generated by python/toy_support_mechanisms.py; do not edit values.
\begin{figure*}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=.44\textwidth,height=5.0cm,
 xlabel={Surviving reference weight-4 words},ylabel={Total $A_4$},
 xmin=-.7,xmax=6.8,ymin=-.7,ymax=7,xtick={0,2,4,6},ytick={0,2,4,6},
 tick label style={font=\small},label style={font=\small},
 title={(a) Complete versus reference-only rejection},title style={font=\small},
 grid=major,grid style={gray!20}]
\addplot[gray,densely dashed,domain=0:6] {x};
\addplot[only marks,mark=*,blue!65!black,mark size=2.7pt,
 nodes near coords,point meta=explicit,
 every node near coord/.append style={font=\scriptsize,anchor=south,yshift=2pt},
 /pgf/number format/fixed,/pgf/number format/1000 sep={{,}}]
 table[x=x,y=y,meta=count] {
x y count
"""+points+r"""
};
\addplot[only marks,mark=o,red!70!black,mark size=4.5pt,thick]
 coordinates {(0,2) (0,4) (0,6)};
\end{axis}
\end{tikzpicture}\hfill
\begin{tikzpicture}
\begin{axis}[width=.44\textwidth,height=5.0cm,
 xlabel={Omitted condition},ylabel={Feasible outer matrices},
 xmin=.35,xmax=7.65,ymin=0,ymax=5000,
 xtick={1,2,3,4,5,6,7},xticklabels={1,2,3,4,5,6,None},
 ytick={0,2048,4096},scaled y ticks=false,tick label style={font=\small},label style={font=\small},
 title={(b) Separate feasibility does not imply compatibility},
 title style={font=\small},ymajorgrids,grid style={gray!20}]
\addplot[ybar,bar width=12pt,fill=blue!30,draw=blue!65!black,
 nodes near coords,every node near coord/.append style={font=\scriptsize},
 /pgf/number format/fixed,/pgf/number format/1000 sep={{,}}]
 coordinates {"""+bars+r"""};
\end{axis}
\end{tikzpicture}
\caption{Exhaustive support-domain checks for
$N=32$ designs with zero internal supports. All outer coefficients are free.
(a) All $2^{14}$ matrices for $\mathcal I_{\rm L}=\{3,5\}$.
Point labels count matrices. Circled points reject every reference weight-4
word but have $A_4>0$. (b) All $2^{21}$ matrices for
$\mathcal I_{\rm L}=\{3,5,7\}$ at target eight.
Both use $\mathcal I_{\rm R}=\{3,5,9,10,11,12,13,14,15\}$.
Condition order is specified in the text. Omitting a condition does not
change the parameter space.}
\label{fig:toy-elimination}
\end{figure*}
""", encoding='ascii')
    data = report['completion']
    exact = coordinates([(v['target'], v['feasible']/data['assignments']) for v in data['targets']])
    bound = coordinates([(v['target'], fraction_value(v['lower_bound'])) for v in data['targets']])
    trace = coordinates([(v['fixed_bits'], fraction_value(v['phi'])) for v in data['trace']])
    directory.joinpath('toy_completion.tex').write_text(r"""% Generated by python/toy_support_mechanisms.py; do not edit values.
\begin{figure*}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[width=.44\textwidth,height=5.0cm,
 xlabel={Trial distance $d_\star$},ylabel={Fraction of outer matrices},
 xmin=3.8,xmax=9.2,ymin=-.04,ymax=1.06,xtick={4,5,6,7,8,9},
 tick label style={font=\small},label style={font=\small},
 title={(a) Exact feasibility and a sufficient bound},title style={font=\small},
 legend style={font=\scriptsize,at={(.98,.98)},anchor=north east},
 grid=major,grid style={gray!20}]
\addplot[blue!65!black,thick,mark=*] coordinates {"""+exact+r"""};
\addlegendentry{Exact feasible fraction}
\addplot[red!70!black,dashed,thick,mark=square*] coordinates {"""+bound+r"""};
\addlegendentry{$[1-\Phi_{\mathcal K}(\varnothing)]_+$}
\end{axis}
\end{tikzpicture}\hfill
\begin{tikzpicture}
\begin{axis}[width=.44\textwidth,height=5.0cm,
 xlabel={Number of fixed outer bits},ylabel={$\Phi_{\mathcal K}(\pi)$},
 xmin=-.3,xmax=14.3,ymin=-.01,ymax=.22,xtick={0,2,4,6,8,10,12,14},
 ytick={0,.0625,.125,.1875},yticklabels={$0$,$1/16$,$1/8$,$3/16$},
 tick label style={font=\small},label style={font=\small},
 title={(b) Conditional-mean completion at $d_\star=6$},
 title style={font=\small},grid=major,grid style={gray!20}]
\addplot[blue!65!black,thick,mark=*] coordinates {"""+trace+r"""};
\end{axis}
\end{tikzpicture}
\caption{Exact outer completion for the $N=32$ constituent pair of
Fig.~\ref{fig:toy-elimination}(a), over all $2^{14}$ outer matrices.
(a) Complete mother-word counts give the feasible fraction.
The lower bound uses all mixed-coset events, together with the pure-right
obstruction when $d_\star>8$. (b) Increasing-source, then increasing-destination
bit order with smaller conditional mean, choosing zero on ties.
These are exact finite counts, not Monte Carlo estimates.}
\label{fig:toy-completion}
\end{figure*}
""", encoding='ascii')
