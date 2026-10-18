"""Sequence model over message events + tabular hybrid for Coordination Referee.

Each message is embedded by (kind, sender, receiver, subtask-bucket, tool-bucket,
refs-flag) and processed with a BiGRU + attention pooling; run-level numeric
features are appended before the classifier head.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import torch
import torch.nn as nn

from engine import parse_run

KINDS = ["assign", "assign_maybe", "deliver", "recovery", "ack", "work", "wait",
         "loop", "drift", "conflict_msg", "state_write", "final", "system", "other"]
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def _bucket(s, n):
    if not s:
        return 0
    return 1 + (int(hashlib.md5(s.encode()).hexdigest(), 16) % (n - 1))


def build_sequence(run, max_len=160):
    ev = run["events"]
    agent_ids = run["agent_ids"]
    aidx = {a: i + 1 for i, a in enumerate(agent_ids)}
    n = len(ev)
    feats = np.zeros((min(n, max_len), 6), dtype=np.int64)
    for i, e in enumerate(ev[:max_len]):
        k = KINDS.index(e["kind"]) if e["kind"] in KINDS else KINDS.index("other")
        feats[i, 0] = k
        feats[i, 1] = aidx.get(e["from"], 0)
        feats[i, 2] = aidx.get(e["to"], 0)
        feats[i, 3] = _bucket(e["sub"], 511)
        feats[i, 4] = _bucket(e["tool"], 127)
        feats[i, 5] = 1 if e["refs"] else 0
    return feats


class SeqModel(nn.Module):
    def __init__(self, n_tabular, hidden=160, layers=2, dropout=0.25):
        super().__init__()
        self.kind = nn.Embedding(len(KINDS), 16, padding_idx=0)
        self.sender = nn.Embedding(14, 12, padding_idx=0)
        self.recv = nn.Embedding(14, 12, padding_idx=0)
        self.sub = nn.Embedding(512, 32, padding_idx=0)
        self.tool = nn.Embedding(128, 12, padding_idx=0)
        self.refse = nn.Embedding(2, 2, padding_idx=0)
        d_in = 16 + 12 + 12 + 32 + 12 + 2
        self.proj = nn.Linear(d_in, hidden)
        self.gru = nn.GRU(hidden, hidden // 2, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(hidden, 1)
        self.tab = nn.Sequential(nn.Linear(n_tabular, 128), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden + 128, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 7))

    def forward(self, seq, tab, mask):
        x = torch.cat([self.kind(seq[..., 0]), self.sender(seq[..., 1]),
                       self.recv(seq[..., 2]), self.sub(seq[..., 3]),
                       self.tool(seq[..., 4]), self.refse(seq[..., 5])], dim=-1)
        x = torch.relu(self.proj(x))
        x, _ = self.gru(x)
        a = self.attn(torch.tanh(x)).squeeze(-1)
        a = a.masked_fill(~mask, -1e4)
        a = torch.softmax(a, dim=-1).unsqueeze(-1)
        pooled = (x * a).sum(1)
        t = self.tab(tab)
        return self.head(torch.cat([pooled, t], dim=-1))


def pad_batch(seqs, device):
    n = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), n, 6), dtype=np.int64)
    mask = np.zeros((len(seqs), n), dtype=bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        mask[i, :len(s)] = True
    return (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))
