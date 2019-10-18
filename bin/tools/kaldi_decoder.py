from __future__ import print_function
from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader


class Kaldi_Decoder(object):
    def __init__(self,beam,max_active,mdl,fst,word,acoustic_scale=0.1):
        decoder_opts = LatticeFasterDecoderOptions()
        decoder_opts.beam = beam
        decoder_opts.max_active = max_active
        #decoder_opts.lattice_beam = 7
        self.asr = MappedLatticeFasterRecognizer.from_files(
            mdl, fst, word,
            acoustic_scale=acoustic_scale, decoder_opts=decoder_opts)

    def decode_feat(self,model,feat,*arg,**args):
        # Decode log-likelihoods represented as numpy ndarrays.
        # Useful for decoding with non-kaldi acoustic models.
        loglikes = model.recognize(feat,*arg,**args)
        out = self.asr.decode(Matrix(loglikes.detach().cpu().numpy()))
        return out

    def decode_loglike(self,loglikes):
        out = self.asr.decode(Matrix(loglikes))
        return out