"""Example 5. Restrict the joint observation matrix and lift a representative."""
from itertools import product
from polar_subcodes.algebra import check_rows, polar_transform, rref, solve, span
from paper_examples.running import code, support


def run():
    checks = check_rows((4,), (0,) * 8)
    rows = []
    for msg in (2, 6):
        u = msg << 3
        inside = support(polar_transform(u, 8))
        columns = [sum(((row >> i) & 1) << k for k, row in enumerate(checks)) for i in inside]
        _, quotient = solve([(col, 0) for col in columns], 7)
        # In column order (t0,t1,t2) each observation row has one block Q
        # for each nonzero bit of this message, and zero blocks elsewhere.
        rows.extend(sum(q << (7 * bit) for bit in support(msg)) for q in quotient)
    assert len(rref(rows)) == 11
    # vec(T) packs seven checks per source column. Selecting these four
    # entries in order (alpha,b0,c,b1) forms the restricted matrix L_3 R.
    coordinates = (0, 7, 8, 14)
    restriction = [sum(((row >> i) & 1) << bit for bit, i in enumerate(coordinates))
                   for row in rows]
    assert restriction.count(0) == 9
    active = [row for row in restriction if row]
    assert active == [0b0100, 0b1110]
    # A feasible observation must be (1,1). The representative (0,0,1,0)
    # gives that pair. The homogeneous system gives all changes preserving it.
    origin, directions = solve([(row, 0) for row in active], 4)
    assert origin == 0 and len(directions) == 2
    representative = 0b0100
    lifted = {tuple((representative ^ d) >> i & 1 for i in range(4)) for d in span(directions)}
    feasible = {config for config in product((0, 1), repeat=4)
                if all(code(*config).encode(msg).bit_count() >= 4 for msg in range(1, 16))}
    assert lifted == feasible
    return {"full_rank": len(rref(rows)), "restricted_rank": len(rref(active)),
            "zero_rows": restriction.count(0), "kernel_directions_(bit_support)": list(map(support, directions)),
            "feasible": sorted(lifted)}


if __name__ == "__main__":
    print(run())
