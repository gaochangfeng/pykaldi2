from kaldi.fstext import CompactLatticeVectorFst,read_fst_kaldi,CompactLatticeConstFst
from kaldi.util.table import SequentialMatrixReader
from kaldi.util.table import CompactLatticeWriter,RandomAccessCompactLatticeReader

rspecifier = "ark:lattice/lat.1"
feats_wspecifier = "ark,scp:my_lat.ark,my_lat.scp"

lat1 = CompactLatticeVectorFst()
lat2 = CompactLatticeVectorFst()

with CompactLatticeWriter(feats_wspecifier) as writer:         
    writer["lat1"] = lat1
    writer["lat2"] = lat2


with RandomAccessCompactLatticeReader(rspecifier) as reader:
    reader["1688-142285-0000"].draw("lattice.dot",width=200, height=1100)