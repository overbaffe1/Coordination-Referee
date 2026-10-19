"""Train the sequence model with 5-fold CV; report OOF Macro F1."""
import sys, time
sys.path.insert(0, 'sol')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from engine import parse_run
from features4 import extract_features
from nn_model import SeqModel, build_sequence, pad_batch, LABELS

torch.manual_seed(42)
np.random.seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train = pd.read_csv('data/train.csv')
t0 = time.time()
seqs, tabs, ys = [], [], []
tab_rows = []
for rec in train.to_dict('records'):
    run = parse_run(rec)
    seqs.append(build_sequence(run))
    tab_rows.append(extract_features(run))
    ys.append(LABELS.index(rec['label']))
X = pd.DataFrame(tab_rows).fillna(0)
X.to_pickle('/tmp/X4b.pkl')
scaler = StandardScaler()
T = scaler.fit_transform(X.values.astype(np.float32))
np.save('/tmp/T.npy', T)
y = np.array(ys)
print('prep', round(time.time() - t0, 1), 'device', DEVICE)

EPOCHS = 14
oof = np.zeros((len(y), 7), dtype=np.float32)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
    model = SeqModel(T.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xtr = [seqs[i] for i in tr]
    Ttr = torch.from_numpy(T[tr]).to(DEVICE)
    ytr = torch.from_numpy(y[tr]).to(DEVICE)
    n = len(tr)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            seq_batch = [Xtr[j] for j in idx]
            xb, mask = pad_batch(seq_batch, DEVICE)
            tb = Ttr[idx]
            yb = ytr[idx]
            logits = model(xb, tb, mask)
            loss = lossf(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sch.step()
    model.eval()
    with torch.no_grad():
        outs = []
        for i in range(0, len(te), 128):
            xb, mask = pad_batch([seqs[j] for j in te[i:i + 128]], DEVICE)
            tb = torch.from_numpy(T[te[i:i + 128]]).to(DEVICE)
            outs.append(torch.softmax(model(xb, tb, mask), -1).cpu().numpy())
        oof[te] = np.concatenate(outs)
    pred = oof[te].argmax(1)
    print(f'fold {fold}: acc={float((pred == y[te]).mean()):.4f} elapsed={time.time()-t0:.0f}s', flush=True)

np.save('/tmp/oof_nn.npy', oof)
p = oof.argmax(1)
print('NN OOF MacroF1:', round(f1_score(y, p, average='macro'), 4))
print(pd.crosstab(y, [LABELS[i] for i in p]))
