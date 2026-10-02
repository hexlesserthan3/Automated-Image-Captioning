"""Inference decoding (owner: Manisha - evaluation & demo)."""
import torch
import torch.nn.functional as F

from common import PAD, START, END


@torch.no_grad()
def greedy(model, feats, max_len=30):
    """Batched greedy decoding. feats: (B, 49, 2048). Returns list of id lists."""
    B, dev = feats.size(0), feats.device
    seq = torch.full((B, 1), START, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_len):
        logits = model(feats, seq)[:, -1]
        logits[:, [PAD, START]] = float("-inf")
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= nxt == END
        if done.all():
            break
    return [s[1:].tolist() for s in seq]


@torch.no_grad()
def beam_search(model, feat, beam=3, max_len=30):
    """Beam search for one image. feat: (1, 49, 2048). Scores are length-normalised
    log-probabilities. Returns one id list."""
    dev = feat.device
    seqs = torch.full((1, 1), START, dtype=torch.long, device=dev)
    scores = torch.zeros(1, device=dev)
    finished = []
    for _ in range(max_len):
        k = seqs.size(0)
        logp = F.log_softmax(model(feat.expand(k, -1, -1), seqs)[:, -1], -1)
        logp[:, [PAD, START]] = float("-inf")
        cand = (scores[:, None] + logp).view(-1)
        top_s, top_i = cand.topk(beam)
        V = logp.size(1)
        keep_seq, keep_sc = [], []
        for s, i in zip(top_s.tolist(), top_i.tolist()):
            b, tok = divmod(i, V)
            new = torch.cat([seqs[b], torch.tensor([tok], device=dev)])
            if tok == END:
                finished.append((s / (len(new) - 1), new[1:].tolist()))
            else:
                keep_seq.append(new)
                keep_sc.append(s)
        if not keep_seq or len(finished) >= beam:
            break
        seqs = torch.stack(keep_seq)
        scores = torch.tensor(keep_sc, device=dev)
    if not finished:
        finished = [(sc / seqs.size(1), sq[1:].tolist()) for sq, sc in zip(seqs, scores.tolist())]
    return max(finished, key=lambda x: x[0])[1]
