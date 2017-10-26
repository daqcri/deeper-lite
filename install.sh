#!/bin/bash
set -e
(
cd torch
./install-deps
./install.sh
. /root/torch/install/bin/torch-activate
luarocks install csvigo
luarocks install dp
)

#GloVe embeddings
(
cd glove
glove=glove.840B.300d.zip
wget http://nlp.stanford.edu/data/$glove
unzip $glove
rm -f $glove
)
