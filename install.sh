#!/bin/bash
set -e
(
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
. ~/torch/install/bin/torch-activate
luarocks install csvigo
luarocks install dp
luarocks install rnn
)

#GloVe embeddings
(
cd glove
glove=glove.840B.300d.zip
wget http://nlp.stanford.edu/data/$glove
unzip $glove
rm -f $glove
)
