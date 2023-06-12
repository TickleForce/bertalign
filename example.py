import os
from bertalign import Bertalign, Encoder
from bertalign.eval import *
from sentence_transformers import SentenceTransformer

src_dir = 'eval/grc'
tgt_dir = 'eval/en'
gold_dir = 'eval/gold'

encoder = Encoder(SentenceTransformer("LaBSE"))

test_alignments = []
gold_alignments = []
for file in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file).replace("\\","/")
    tgt_file = os.path.join(tgt_dir, file).replace("\\","/")
    src = open(src_file, encoding='utf-8').read()
    tgt = open(tgt_file, encoding='utf-8').read()

    aligner = Bertalign(src, tgt, encoder, show_logs=True)
    aligner.align_sents()
    #aligner.print_sents()

    test_alignments.append(aligner.result)

    gold_file = os.path.join(gold_dir, file)
    gold_alignments.append(read_alignments(gold_file))

scores = score_multiple(gold_list=gold_alignments, test_list=test_alignments)
log_final_scores(scores)
