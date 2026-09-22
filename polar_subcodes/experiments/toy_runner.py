"""Reproduce the N32 complete-support experiments and editable figure data."""
from fractions import Fraction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polar_subcodes.io import Progress,write_json
from .toys import outer_directory,RIGHT,replacement,overlap,completion,figures


def run(output):
    emit = Progress(output)
    emit('toy','Enumerate all 16,384 assignments of the 14-bit outer domain')
    directory = outer_directory((3,5),RIGHT)
    report = dict(sampling='exhaustive, no random seeds',replacement=replacement(directory),
                  completion=completion(directory))
    emit('toy','Check overlapping conditions against all 2,097,152 outer assignments')
    report['overlap'] = overlap()
    assert report['replacement']['false_eliminations']==2530
    assert report['replacement']['true_eliminations']==13472
    assert report['overlap']['all_conditions']==0
    assert [r['feasible'] for r in report['completion']['targets']]==[16384,13472,13472,1536,1536,0]
    write_json(emit.folder/'report.json',report)
    figures(report,emit.folder/'pgfplots')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    data = report['replacement']['histogram']
    points = axes[0].scatter([r['old_survivors'] for r in data],[r['total_a4'] for r in data],
        s=[12+2*r['supports']**0.5 for r in data],c=[r['supports'] for r in data],cmap='viridis')
    axes[0].set(xlabel='Surviving old weight-4 words',ylabel='All candidate weight-4 words',title='Complete mother versus old words')
    fig.colorbar(points,ax=axes[0],label='Support assignments')
    data = report['overlap']['leave_one_out']
    axes[1].bar([str(r['omitted']) for r in data]+['None'],[r['feasible'] for r in data]+[0],color='#35788a')
    axes[1].set(xlabel='Omitted condition',ylabel='Feasible assignments',title='Joint compatibility')
    fig.savefig(emit.folder/'elimination.png',dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    data = report['completion']['targets']
    axes[0].plot([r['target'] for r in data],[r['feasible']/16384 for r in data],'o-',label='Exact feasible fraction')
    axes[0].plot([r['target'] for r in data],[float(Fraction(**r['lower_bound'])) for r in data],'s--',label='Conditional-mean lower bound')
    axes[0].set(xlabel='Trial distance',ylabel='Fraction',title='Feasibility is target dependent')
    axes[0].legend()
    data = report['completion']['trace']
    axes[1].step([r['fixed_bits'] for r in data],[float(Fraction(**r['phi'])) for r in data],where='post')
    axes[1].set(xlabel='Fixed support coefficients',ylabel='Expected forbidden events',title='Deterministic completion')
    fig.savefig(emit.folder/'completion.png',dpi=180)
    plt.close(fig)
    emit('toy','Every event and conditional mean agrees with exhaustive counting')
    return report
