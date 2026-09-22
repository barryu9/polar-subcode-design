"""Sequentially rerun every design and report timing variation."""
from _entry import run
if __name__=='__main__':
    run(['reproduce','all','--repetitions','3'])
