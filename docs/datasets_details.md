# This document contains a summary for the SNLI and SST data sets
## Experiment Global Vocabulary
Experiment global vocabulary is a word set that contains all the words in the SST and SNLI data sets.
Pickled global vocabulary is in the '/datasets/Diy/etc/dict.pkl' file.

## Glove 840B word embedding
We use glove 840B word embedding as our word embedding.


## SST data set
Number of 'unk' tokens (words not included in 6B glove embedding) in the whole set
Total: 2048
Number of 'unk' tokens (words not included in 840B glove embedding) in the whole set
Total: 1767

## SNLI data set
Total sentences pairs in each set:

Number of sentence pairs in each set with no glod-label:
test: 176
dev: 158
train: 785

Number of 'unk' tokens (words not included in 6B glove embedding)
test: 640
dev: 600
train: 30255

Number of 'unk' tokens (words not included in 840B glove embedding)
test: 259
dev: 263
train: 13664

Much less unseen words when using 840B Glove.