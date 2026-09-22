"""Seeded BPSK-AWGN Monte Carlo simulation for arbitrary saved supports."""
import math
import time
import numpy as np

from polar_subcodes.io import Progress, write_json, hardware
from .scl import decode, full_scl_backend_name


def wilson(errors,frames):
    """Two-sided 95% Wilson interval for a fixed frame count."""
    z = 1.959963984540054
    p = errors/frames
    denominator = 1+z*z/frames
    center = (p+z*z/(2*frames))/denominator
    width = z*math.sqrt(p*(1-p)/frames+z*z/(4*frames*frames))/denominator
    return [max(0,center-width),min(1,center+width)]


def simulate(code,output,ebn0_db=3.0,frames=1000,list_size=64,batch_size=32,
             seed=2026,backend='auto',messages='zero'):
    if not 0<code.k<=code.n or type(frames) is not int or frames<1 or batch_size<1:
        raise ValueError('Positive K, frames and batch size are required')
    if not math.isfinite(ebn0_db) or messages not in ('zero','random'):
        raise ValueError('Use a finite Eb/N0 and zero or random messages')
    if type(seed) is not int or seed<0:
        raise ValueError('Seed must be a nonnegative integer')
    emit = Progress(output)
    noise_seed,message_seed = np.random.SeedSequence(seed).spawn(2)
    noise_rng = np.random.default_rng(noise_seed)
    message_rng = np.random.default_rng(message_seed)
    variance = 1/(2*(code.k/code.n)*10**(ebn0_db/10))
    write_json(emit.folder/'support.json',code.as_dict())
    write_json(emit.folder/'configuration.json',dict(N=code.n,K=code.k,ebn0_db=ebn0_db,
        noise_variance=variance,frames=frames,list_size=list_size,batch_size=batch_size,
        seed=seed,backend=full_scl_backend_name(backend),messages=messages,
        channel='BPSK AWGN, sigma^2=1/(2 R Eb/N0)',decoder='min-sum SCL, no CRC',
        stopping_rule='fixed number of frames',hardware=hardware()))
    completed = errors = bit_errors = 0
    last = time.perf_counter()
    emit('simulation','Compare the actual decoded information bits with transmitted bits',
         EbN0_dB=ebn0_db,L=list_size,frames=frames,backend=full_scl_backend_name(backend))
    while completed<frames:
        batch = min(batch_size,frames-completed)
        # Separate streams make noise independent of the message-generation option.
        bits = (message_rng.random((batch,code.k))<0.5).astype(np.int8) if messages=='random' else np.zeros((batch,code.k),dtype=np.int8)
        if messages=='zero':
            transmitted = np.zeros((batch,code.n),dtype=np.int8)
        else:
            encoded = [code.encode(sum(int(bit)<<q for q,bit in enumerate(row))) for row in bits]
            transmitted = np.array([[(x>>i)&1 for i in range(code.n)] for x in encoded],dtype=np.int8)
        received = 1-2*transmitted+noise_rng.normal(0,math.sqrt(variance),(batch,code.n))
        decoded = decode(code,2*received/variance,list_size,backend)
        bad = decoded[:,code.profile]!=bits
        errors += int(np.count_nonzero(np.any(bad,axis=1)))
        bit_errors += int(np.count_nonzero(bad))
        completed += batch
        now = time.perf_counter()
        if completed==frames or now-last>=5:
            report = dict(frames=completed,frame_errors=errors,FER=errors/completed,
                          information_bit_errors=bit_errors,BER=bit_errors/(completed*code.k),
                          FER_95_percent_interval=wilson(errors,completed),
                          seconds=now-emit.start,complete=completed==frames)
            write_json(emit.folder/'report.json',report)
            emit('simulation','Measured errors with uncertainty, zero errors do not mean zero FER',
                 frames=completed,errors=errors,FER=report['FER'],interval=report['FER_95_percent_interval'])
            last = now
    return report
