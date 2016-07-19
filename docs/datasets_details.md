# Summary of the SNLI and SST data sets for experiments
## Experiment Global Vocabulary
Experiment global vocabulary is a word set that contains all the words in the SST and SNLI data sets.
Pickled global vocabulary is in the `/datasets/Diy/etc/dict.pkl` file.

## Glove 840B word embedding
We use glove 840B word embedding as word embedding.

## SST data set
Number of `'<unk>'` tokens (words not included in glove embedding) in the whole set: 2048(6B glove)|1767(840 glove)

## SNLI data set
Total sentences pairs in each set:
Before clean:
test/dev/train 10,000/10,000/55,0152
After clean:
test/dev/train 9,824/9,842/549367

Distinct words: 73,026


Number of sentence pairs in each set with no glod-label:
test/dev/train 176/158/785

Number of `'<unk>'` tokens (words not included in 6B glove embedding)
test/dev/train 640/600/30,255

Number of `'<unk>'` tokens (words not included in 840B glove embedding)
test/dev/train 259/263/13,664

Much less unseen words when using 840B Glove.