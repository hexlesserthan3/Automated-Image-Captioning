# Automated Image Captioning: Retrieval vs LSTM vs Transformer on Flickr8k

Deep Learning Project (ICT-4442), Manipal Institute of Technology, Manipal
Anikait Chitlangia (230911256) · P Manisha (230911106) · Siya Rath (230911050)

## Overview

This project compares three image-captioning models from different families under one common protocol on Flickr8k:

1. **Retrieval baseline** (no training): returns a consensus caption from the k most similar training images.
2. **LSTM decoder**: a Show-and-Tell style decoder initialised from a mean-pooled image vector.
3. **Transformer decoder**: cross-attends over the full 7x7 ResNet-50 feature grid.

All models share the same split, vocabulary, frozen ResNet-50 features and metric code (BLEU-1 to 4, METEOR, CIDEr against all five references). Hyperparameters were tuned on the validation set only, and the test set was evaluated once. The project also compares greedy decoding with beam search and includes a hand-labelled error analysis.

## Results (test set, 1,000 images)

| Model | Decoding | BLEU-4 | METEOR* | CIDEr | Distinct captions |
|---|---|---|---|---|---|
| Retrieval | k = 30 | 0.153 | 0.381 | 0.408 | 521 |
| LSTM | greedy | 0.170 | 0.385 | 0.412 | 799 |
| LSTM | beam 3 | 0.191 | 0.406 | 0.465 | 746 |
| Transformer | greedy | 0.189 | 0.403 | 0.479 | 941 |
| Transformer | beam 3 | **0.204** | **0.419** | **0.526** | 917 |

\*NLTK implementation, not the official Java METEOR, so values are not comparable with published papers.

Main findings:
- The Transformer with beam search (width 3) is best on every metric. Beam search improved every metric for both neural models at about 1.3 to 1.5 times the decoding time of greedy search, but reduced caption diversity.
- The retrieval baseline, which needs no training, reaches a CIDEr about level with the greedy LSTM.
- The Transformer overfits after epoch 6. Failure analysis shows mainly missed or substituted subjects, wrong activities, hallucinated objects and generic template captions.

The full analysis is in the project report.

## Repository structure

| File | Purpose |
|---|---|
| `common.py` | Caption cleaning, split, vocabulary, dataset and collate functions |
| `extract_features.py` | Frozen ResNet-50 feature extraction and caching (7x7x2048 grid) |
| `models.py` | LSTM and Transformer decoders |
| `train.py` | Training loop (teacher forcing, AdamW, early stopping) |
| `decode.py` | Greedy and beam-search decoding |
| `evaluate.py` | BLEU, METEOR, CIDEr, latency, repetition statistics, per-image caption CSV |
| `knn_baseline.py` | Retrieval baseline |
| `error_analysis.py` | Comparison table, failure statistics, label sheets and tally |
| `label_failures.py` | Click-to-label tool for the failure sheets (Colab) |
| `requirements.txt` | Python dependencies |

Cached features and model checkpoints are not stored in this repository.

## Setup and reproduction

Run on a GPU runtime (the experiments used Google Colab). Download Flickr8k from Kaggle (`adityajn105/flickr8k`), which gives an `Images/` folder and `captions.txt`. Set `ROOT` to that folder and `WORK` to an output folder for features, checkpoints and results.

```bash
pip install -r requirements.txt

# 1. Cache frozen ResNet-50 features
python extract_features.py --root $ROOT --work $WORK

# 2. Train the two neural models (seed 42)
python train.py --model lstm --root $ROOT --work $WORK
python train.py --model transformer --root $ROOT --work $WORK

# 3. Tune beam width on validation, then evaluate on test once
python evaluate.py --model transformer --split val --beams 3,5 --root $ROOT --work $WORK
python evaluate.py --model transformer --split test --beams 3 --root $ROOT --work $WORK
python evaluate.py --model lstm --split test --beams 3 --root $ROOT --work $WORK

# 4. Retrieval baseline: tune k on validation, then evaluate on test once
python knn_baseline.py --split val --ks 1,3,5,10,15,20,30 --root $ROOT --work $WORK
python knn_baseline.py --split test --ks 30 --root $ROOT --work $WORK

# 5. Comparison table and error analysis
python error_analysis.py table --work $WORK
python error_analysis.py sample --work $WORK --n 50   # writes label sheets (never overwrites labelled ones)
python error_analysis.py tally --work $WORK           # after labelling
```

Notes:
- Re-running the same command with the same seed reproduced identical losses and metrics in our runs.
- Each test run of `knn_baseline.py` overwrites `knn_test_metrics.json`, so run it once with the final k.
- Label sheets are filled with `label_failures.py` in a Colab notebook: `from label_failures import label; label("lstm", ROOT, WORK)`.

## Method summary

- **Data:** Flickr8k, 8,091 images with 5 captions each. Captions are lowercased, stripped of punctuation and truncated to 30 tokens. The vocabulary (2,570 words) is built from training captions only, with words below frequency 5 mapped to `<unk>`. The split is 6,000 / 1,000 / 1,000 (seeded random, seed 42).
- **Encoder:** ImageNet-pretrained ResNet-50, frozen, features taken before global pooling.
- **LSTM:** one layer, 512 units, initial state from the mean-pooled image vector (6.8 M parameters).
- **Transformer:** 3 pre-norm decoder layers, 8 heads, d = 512, causal self-attention plus cross-attention over 49 grid cells (13.2 M parameters).
- **Retrieval:** cosine similarity over mean-pooled features, candidate captions pooled from the k nearest training images, consensus selected by unigram and bigram F1.
- **Decoding:** greedy, and beam search with length-normalised log-probability.

## Limitations

- The split is a seeded random split, not the Karpathy split, so results are not directly comparable with published Flickr8k numbers.
- All results come from a single seed, without confidence intervals.
- METEOR is NLTK's implementation. CIDEr was computed without stemming or the PTB tokenizer, so singular and plural forms count as different words.
- Failure labels were assigned by hand by different people and are subjective, and image properties were labelled only for low-scoring captions, so no causal claim is made.
- Only the beam width and the retrieval neighbour count were tuned. The encoder was frozen throughout.
- Decoding times come from a simple implementation without key/value caching. The retrieval step runs unoptimised on CPU, so its time is not comparable with the GPU decoders.

## Team and contributions

The individual contribution statement is in Appendix A of the project report, and the commit history of this repository records who committed what.

## References

1. Vinyals et al., "Show and Tell: A Neural Image Caption Generator," CVPR 2015.
2. Xu et al., "Show, Attend and Tell," ICML 2015.
3. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
4. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
5. Karpathy and Fei-Fei, "Deep Visual-Semantic Alignments for Generating Image Descriptions," CVPR 2015.
6. Hodosh, Young and Hockenmaier, "Framing Image Description as a Ranking Task," JAIR 2013 (Flickr8k).
7. Devlin et al., "Exploring Nearest Neighbor Approaches for Image Captioning," 2015.
8. Papineni et al. (BLEU, 2002); Banerjee and Lavie (METEOR, 2005); Vedantam et al. (CIDEr, CVPR 2015).

The Flickr8k dataset is not included. Please follow its terms of use.
