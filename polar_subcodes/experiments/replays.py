"""Sequential repeated constructions with identical-support checks."""
import statistics
from polar_subcodes.io import Progress,write_json,hardware
from .reproduce import reproduce


def run(output,names,repetitions,limits):
    emit=Progress(output)
    report=dict(repetitions=repetitions,hardware=hardware(),sequential=True,trials=[])
    for index in range(repetitions):
        trial=[]
        for name in names:
            emit('replay','Run a fresh zero-support construction',case=name,repetition=index+1)
            data=reproduce(name,emit.folder/f'trial_{index+1}'/name,limits)
            trial.append(dict(case=name,seconds=data['total_seconds'],dmin=data['dmin'],
                              Admin=data['Admin'],support_matches_archive=True))
        report['trials'].append(trial)
        report['summary']={name:dict(median_seconds=statistics.median(
            next(case['seconds'] for case in t if case['case']==name) for t in report['trials']),
            seconds=[next(case['seconds'] for case in t if case['case']==name) for t in report['trials']])
            for name in names}
        write_json(emit.folder/'report.json',report)
    return report
