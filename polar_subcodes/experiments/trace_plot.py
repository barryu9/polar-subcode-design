"""Plot exact conditional-mean paths without manuscript-dependent numbering."""
from fractions import Fraction
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    folder=Path(folder)
    frames=json.loads((folder/'outer/frontier.json').read_text())['frames']
    selected=[f for f in frames if f.get('conditional_mean_path') and
              Fraction(f['conditional_mean_path'][0]['expectation'])>=1]
    if not selected:
        selected=frames[-3:]
    fig,axis=plt.subplots(figsize=(6.4,3.4),layout='constrained')
    for frame in selected:
        points=frame['conditional_mean_path']
        axis.step([p['fixed_bits'] for p in points],
                  [float(Fraction(p['expectation'])) for p in points],
                  where='post',label=f"source {frame['source']}")
    axis.axhline(1,color='0.4',linestyle='--',linewidth=1)
    axis.set(xlabel='Fixed coefficients in the current source column',ylabel='Conditional mean of forbidden events')
    axis.legend()
    fig.savefig(folder/'conditional_means.png',dpi=180)
    plt.close(fig)
    print(folder/'conditional_means.png')
