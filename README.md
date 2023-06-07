# Bertalign for Ancient Greek

This repo is a fork of [Bertalign](https://github.com/bfsujason/bertalign), a mulitlingual sentence aligner, updated for aligning ancient Greek texts with English translations.

Bertalign is designed to facilitate the construction of multilingual parallel corpora and translation memories, which have a wide range of applications in translation-related research such as corpus-based translation studies, contrastive linguistics, computer-assisted translation, translator education and machine translation.

## Approach

Bertalign uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) to represent source and target sentences so that semantically similar sentences in different languages are mapped onto similar vector spaces. Then a two-step algorithm based on dynamic programming is performed: 1) Step 1 finds the 1-1 alignments for approximate anchor points; 2) Step 2 limits the search path to the anchor points and extracts all the valid alignments with 1-many, many-1 or many-to-many relations between the source and target sentences.

## Performance

The gold alignment dataset is based on translations of the Didache, letter of Polycarp, a Greek reader, and works of Josephus.

LaBSE on eval dataset:
```
 ---------------------------------
|             |  Strict |    Lax  |
| Precision   |   0.946 |   0.999 |
| Recall      |   0.935 |   1.000 |
| F1          |   0.941 |   1.000 |
 ---------------------------------
```

Using a sentence transformer trained on parallel data: 
```
 ---------------------------------
|             |  Strict |    Lax  |
| Precision   |   0.970 |   0.994 |
| Recall      |   0.956 |   1.000 |
| F1          |   0.963 |   0.997 |
 ---------------------------------
```

## Installation

Please see [requirements.txt](./requirements.txt) for installation. 

## Basic example

See `example.py`.

## Citation

Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, *Digital Scholarship in the Humanities*. [https://doi.org/10.1093/llc/fqac089](https://doi.org/10.1093/llc/fqac089).

## Licence

Bertalign is released under the [GNU General Public License v3.0](./LICENCE)

## Credits

##### Main Libraries

* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

* [faiss](https://github.com/facebookresearch/faiss)

* [sentence-splitter](https://github.com/mediacloud/sentence-splitter)

##### Other Sentence Aligners

* [Hunalign](http://mokk.bme.hu/en/resources/hunalign/)

* [Bleualign](https://github.com/rsennrich/Bleualign)

* [Vecalign](https://github.com/thompsonb/vecalign)