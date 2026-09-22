"""File-based exact spectrum and complete-word interfaces."""
from collections import Counter

from polar_subcodes.algebra import polar_transform
from polar_subcodes.io import Progress,write_json
from .backend import count,words
from .decomposition import spectrum
from .quotient import evaluate


def enumerate_code(code,cutoff,output,method='decomposition',max_work=100_000_000):
    if type(cutoff) is not int or not 0<=cutoff<=code.n or max_work<1:
        raise ValueError('Invalid cutoff or work limit')
    emit = Progress(output)
    emit('enumeration','Only a finished computation is labeled exact',N=code.n,K=code.k,cutoff=cutoff,method=method)
    directory = None
    try:
        if method=='decomposition':
            histogram = spectrum(code,cutoff,max_work,emit,emit.folder/'stages')
        elif method=='words':
            directory,metadata = words(code,cutoff,max_work)
            histogram = metadata['spectrum']
        elif method in ('native','python'):
            histogram = count(code,cutoff,max_work,'cpp' if method=='native' else 'python')['spectrum']
        elif method in ('quotient','direct'):
            values,metadata = evaluate(dict(kind='directory',n=code.n,checks=code.checks,cutoff=cutoff),method,max_work)
            directory = [(u,polar_transform(u,code.n).bit_count()) for u in values if u]
            histogram = {0:1,**dict(Counter(w for _,w in directory))}
        else:
            raise ValueError('Unknown enumeration method')
        if directory is not None:
            with (emit.folder/'words.jsonl').open('w',encoding='utf-8') as stream:
                import json
                for u,weight in directory:
                    stream.write(json.dumps(dict(input=str(u),codeword=str(polar_transform(u,code.n)),weight=weight))+'\n')
        positive = [w for w,a in histogram.items() if w and a]
        result = dict(exact=True,N=code.n,K=code.k,cutoff=cutoff,method=method,spectrum=histogram,
                      dmin=min(positive) if positive else None,distance_lower_bound=min(positive) if positive else cutoff+1,
                      word_list_complete_through_cutoff=directory is not None,zero_word_omitted_from_list=True)
        write_json(emit.folder/'result.json',result)
        emit('enumeration','Complete bounded-weight result',spectrum=histogram)
        return result
    except Exception as error:
        write_json(emit.folder/'result.json',dict(exact=False,error=str(error),method=method,cutoff=cutoff))
        raise
