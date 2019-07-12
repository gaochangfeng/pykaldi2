# pykaldi2

PyKaldi2 is a speech toolkit that is built based on [Kaldi](http://kaldi-asr.org/) and [PyTorch](https://pytorch.org/). It relies on [PyKaldi](https://github.com/pykaldi/pykaldi) -- the Python wrapper of Kaldi, to access Kaldi functionalities. The key features of PyKaldi2 are one-the-fly lattice generation for lattice-based sequence training, on-the-fly data simulation and on-the-fly alignment gereation. 

## How to install

PyKaldi2 runs on top of [Horovod](https://github.com/horovod/horovod) and PyKaldi libraries. The dockerfile is provided to customarize the envriorment. To use the repo, do the following three steps. 

1. Clone the repo by

   ```
     git clone https://github.com/jzlianglu/pykaldi2.git
   ```
2. Build the docker image, simply run

  ```
    docker build -t horovod-pykaldi -f docker/dockerfile 
  ```

3. Activate the docker image, for example

  ```
    NV_GPU=0,1,2,3 nvidia-docker run -v `pwd`:`pwd` -w `pwd` --shm-size=32G -i -t horovod-pykaldi
  ```

If you want to run multi-GPU jobs using Horovod, command is like

  ```
    horovodrun -np 4 -H localhost:4 sh run_ce.sh 
  ```

## Cross-entropy training

An example of runing a cross-entropy job is

  ```
   python train_ce.py -config configs/ce.yaml \  
    -data configs/data.yaml \                 
    -exp_dir exp/tr960_blstm_3x512_dp02 \     
    -lr 0.0002 \                              
    -batch_size 64 \                          
    -sweep_size 960 \                         
    -aneal_lr_epoch 3 \                       
    -num_epochs 8 \                           
    -aneal_lr_ratio 0.5                 
  ```

## Sequence training

An example of runing a sequence traing job is
 
  ```   
python train_se.py -config configs/mmi.yaml \
    -data configs/data.yaml \
    -exp_dir exp/tr960_blstm_3x512_dp02 \
    -criterion "mmi" \
    -seed_model exp/tr960_blstm_3x512_dp02/model.7.tar \
    -prior_path /datadisk2/lial/librispeech/s5/exp/tri6b_ali_tr960/final.occs \
    -trans_model /datadisk2/lial/librispeech/s5/exp/tri6b_ali_tr960/final.mdl \
    -den_dir /datadisk2/lial/librispeech/s5/exp/tri6b_denlats_960_cleaned/dengraph \
    -lr 0.000001 \
    -ce_ratio 0.1 \
    -max_grad_norm 5 \
    -batch_size 4 \
  ```

## Reference

Liang Lu, Xiong Xiao, Zhuo Chen, Yifan Gong, "PyKaldi2: Yet another speech toolkit based on Kaldi and PyTorch", arxiv, 2019
