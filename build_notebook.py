import json

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# install packages
cells.append(code([
    "!pip install -q timm grad-cam albumentations kaggle ipywidgets\n",
]))

# imports and setup
cells.append(code("""\
import os, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
""".splitlines(keepends=True)))

# mount drive
cells.append(code("""\
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/DeepGuard-AI')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print('Drive mounted at', DRIVE_DIR)
except Exception:
    DRIVE_DIR = Path('/content/DeepGuard-AI')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print('No drive, saving to', DRIVE_DIR)

CKPT_PATH = DRIVE_DIR / 'best_model.pth'
""".splitlines(keepends=True)))

# kaggle credentials
cells.append(code("""\
from google.colab import files
print('Upload kaggle.json:')
uploaded = files.upload()
kaggle_dir = Path('/root/.kaggle')
kaggle_dir.mkdir(exist_ok=True)
with open(kaggle_dir / 'kaggle.json', 'wb') as f:
    f.write(list(uploaded.values())[0])
os.chmod(kaggle_dir / 'kaggle.json', 0o600)
""".splitlines(keepends=True)))

# download dataset
cells.append(code([
    "!kaggle datasets download -d kshitizbhargava/deepfake-face-images -p /content/ --unzip\n"
]))

# find folders
cells.append(code("""\
for p in sorted(Path('/content').rglob('*')):
    if p.is_dir():
        imgs = list(p.glob('*.jpg')) + list(p.glob('*.png')) + list(p.glob('*.jpeg'))
        if len(imgs) > 5:
            print(len(imgs), '->', p)
""".splitlines(keepends=True)))

# detect real/fake dirs
cells.append(code("""\
# if this fails, set REAL_DIR and FAKE_DIR manually based on the output above
REAL_DIR, FAKE_DIR = None, None
for folder in sorted(Path('/content').rglob('*')):
    if not folder.is_dir(): continue
    n = folder.name.lower()
    imgs = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg'))
    if len(imgs) < 10: continue
    if 'real' in n and REAL_DIR is None: REAL_DIR = folder
    if ('fake' in n or 'stylegan' in n or 'synthetic' in n) and FAKE_DIR is None: FAKE_DIR = folder

print('Real dir:', REAL_DIR)
print('Fake dir:', FAKE_DIR)

real_paths = list(REAL_DIR.glob('*.jpg')) + list(REAL_DIR.glob('*.png')) + list(REAL_DIR.glob('*.jpeg'))
fake_paths = list(FAKE_DIR.glob('*.jpg')) + list(FAKE_DIR.glob('*.png')) + list(FAKE_DIR.glob('*.jpeg'))
print('Real:', len(real_paths), ' Fake:', len(fake_paths))
""".splitlines(keepends=True)))

# build dataframe
cells.append(code("""\
df_real = pd.DataFrame({'path': [str(p) for p in real_paths], 'label': 0, 'class': 'Real'})
df_fake = pd.DataFrame({'path': [str(p) for p in fake_paths], 'label': 1, 'class': 'Fake'})
df = pd.concat([df_real, df_fake], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(df['class'].value_counts())
df.head()
""".splitlines(keepends=True)))

# EDA - class distribution
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0d1117')
counts = df['class'].value_counts()
pal = ['#4CAF50', '#F44336']

ax = axes[0]
ax.set_facecolor('#161b22')
bars = ax.bar(counts.index, counts.values, color=pal, edgecolor='white', width=0.5)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 50, f'{v:,}', ha='center', color='white', fontweight='bold')
ax.set_title('Class Distribution', color='white', fontsize=14, fontweight='bold')
ax.set_xlabel('Class', color='#aaa'); ax.set_ylabel('Count', color='#aaa')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor('#333')

ax2 = axes[1]
ax2.set_facecolor('#161b22')
ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=pal, startangle=90,
        textprops={'color': 'white', 'fontsize': 12}, wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
ax2.set_title('Class Split', color='white', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(str(DRIVE_DIR / 'class_distribution.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
""".splitlines(keepends=True)))

# EDA - sample grid
cells.append(code("""\
def show_sample_grid(df, n=8):
    fig, axes = plt.subplots(2, n, figsize=(18, 8))
    fig.patch.set_facecolor('#0d1117')
    cls_colors = {'Real': '#4CAF50', 'Fake': '#F44336'}
    for r, cls in enumerate(['Real', 'Fake']):
        samp = df[df['class'] == cls].sample(n, random_state=SEED)
        for c, (_, row) in enumerate(samp.iterrows()):
            ax = axes[r, c]
            ax.imshow(Image.open(row['path']).resize((128, 128)))
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_edgecolor(cls_colors[cls]); sp.set_linewidth(2.5); sp.set_visible(True)
            if c == 0:
                ax.set_ylabel(cls, color=cls_colors[cls], fontsize=12, fontweight='bold', rotation=90, labelpad=6)
    plt.tight_layout()
    plt.savefig(str(DRIVE_DIR / 'sample_grid.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()

show_sample_grid(df)
""".splitlines(keepends=True)))

# EDA - pixel distribution
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')
ch_names = ['Red', 'Green', 'Blue']
ch_colors = ['#ef5350', '#66bb6a', '#42a5f5']
s_real = df[df['class'] == 'Real'].sample(150, random_state=SEED)
s_fake = df[df['class'] == 'Fake'].sample(150, random_state=SEED)

for ci in range(3):
    ax = axes[ci]; ax.set_facecolor('#161b22')
    for sub, lbl, ls in [(s_real, 'Real', '-'), (s_fake, 'Fake', '--')]:
        vals = np.concatenate([np.array(Image.open(r['path']).convert('RGB').resize((64, 64)))[:, :, ci].flatten()
                               for _, r in sub.iterrows()])
        ax.hist(vals, bins=64, density=True, alpha=0.65, color=ch_colors[ci], histtype='stepfilled', linestyle=ls, label=lbl)
    ax.set_title(ch_names[ci], color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pixel Value', color='#aaa'); ax.tick_params(colors='white')
    ax.legend(facecolor='#0d1117', labelcolor='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

plt.suptitle('Pixel Intensity - Real vs Fake', color='white', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(str(DRIVE_DIR / 'pixel_distribution.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
""".splitlines(keepends=True)))

# hyperparams and split
cells.append(code("""\
IMG_SIZE = 380
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 avoids multiprocessing issues in Colab
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
LABEL_SMOOTH = 0.1
DROPOUT = 0.4

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)
print(f'Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}')
""".splitlines(keepends=True)))

# transforms and dataset
cells.append(code("""\
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tfm = T.Compose([
    T.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(0.5),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
    T.RandomGrayscale(0.1),
    T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.RandomApply([T.RandomRotation(15)], p=0.3),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])
val_tfm = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

class DeepfakeDataset(Dataset):
    def __init__(self, df, tfm=None):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['path']).convert('RGB')
        return (self.tfm(img) if self.tfm else img), int(row['label'])

train_ds = DeepfakeDataset(train_df, train_tfm)
val_ds   = DeepfakeDataset(val_df,   val_tfm)
test_ds  = DeepfakeDataset(test_df,  val_tfm)

# weighted sampler to handle class imbalance
cc = train_df['label'].value_counts().sort_index().values
cw = 1.0 / torch.tensor(cc, dtype=torch.float)
sw = cw[train_df['label'].values]
sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS, pin_memory=True)
print(f'Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}')
""".splitlines(keepends=True)))

# model
cells.append(code("""\
class DeepGuardNet(nn.Module):
    def __init__(self, num_classes=2, dropout=DROPOUT, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat), nn.Dropout(dropout),
            nn.Linear(feat, 512), nn.GELU(),
            nn.BatchNorm1d(512),  nn.Dropout(dropout * 0.75),
            nn.Linear(512, num_classes),
        )
        self._freeze(0.7)

    def _freeze(self, ratio):
        params = list(self.backbone.parameters())
        n = int(len(params) * ratio)
        for i, p in enumerate(params): p.requires_grad = (i >= n)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True

    def forward(self, x): return self.head(self.backbone(x))

model = DeepGuardNet().to(DEVICE)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total/1e6:.2f}M   Trainable: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)')
""".splitlines(keepends=True)))

# loss / optimizer / scheduler
cells.append(code("""\
class LabelSmoothCE(nn.Module):
    def __init__(self, s=0.1): super().__init__(); self.s = s
    def forward(self, p, t):
        n = p.size(-1); lp = F.log_softmax(p, dim=-1)
        return (1 - self.s) * F.nll_loss(lp, t) + self.s * (-lp.sum(-1).mean() / n)

criterion = LabelSmoothCE(LABEL_SMOOTH)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
""".splitlines(keepends=True)))

# train/eval functions
cells.append(code("""\
def train_epoch(model, loader, opt, crit, scaler, dev):
    model.train(); loss_sum = correct = total = 0
    for imgs, labs in tqdm(loader, desc='train', leave=False):
        imgs, labs = imgs.to(dev), labs.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(dev.type == 'cuda')):
            out = model(imgs); loss = crit(out, labs)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labs).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval(); loss_sum = correct = total = 0
    probs_all = []; labs_all = []
    for imgs, labs in tqdm(loader, desc='eval', leave=False):
        imgs, labs = imgs.to(dev), labs.to(dev)
        with torch.cuda.amp.autocast(enabled=(dev.type == 'cuda')):
            out = model(imgs); loss = crit(out, labs)
        loss_sum += loss.item() * imgs.size(0)
        p = torch.softmax(out, 1)[:, 1]
        correct += (out.argmax(1) == labs).sum().item(); total += imgs.size(0)
        probs_all.extend(p.cpu().numpy()); labs_all.extend(labs.cpu().numpy())
    auc = roc_auc_score(labs_all, probs_all)
    return loss_sum / total, correct / total, auc, np.array(probs_all), np.array(labs_all)
""".splitlines(keepends=True)))

# training loop
cells.append(code("""\
history = {'tl': [], 'vl': [], 'ta': [], 'va': [], 'auc': [], 'lr': []}
best_auc = 0.0; pat = 0; UNFREEZE = 8

for ep in range(1, EPOCHS + 1):
    if ep == UNFREEZE + 1:
        model.unfreeze_all()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)
        print(f'Epoch {ep}: unfreezing full backbone')

    tl, ta = train_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
    vl, va, auc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()
    cur_lr = optimizer.param_groups[0]['lr']

    history['tl'].append(tl); history['vl'].append(vl)
    history['ta'].append(ta); history['va'].append(va)
    history['auc'].append(auc); history['lr'].append(cur_lr)

    mark = ' *' if auc > best_auc else ''
    print(f'[{ep:02d}/{EPOCHS}] train loss:{tl:.4f} acc:{ta:.4f} | val loss:{vl:.4f} acc:{va:.4f} auc:{auc:.4f}{mark}')

    if auc > best_auc:
        best_auc = auc; pat = 0
        torch.save({'epoch': ep, 'state_dict': model.state_dict(), 'val_auc': auc, 'val_acc': va}, str(CKPT_PATH))
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f'Early stopping at epoch {ep}'); break

print(f'Best val AUC: {best_auc:.4f}')
""".splitlines(keepends=True)))

# training curves
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.patch.set_facecolor('#0d1117')
ep_range = range(1, len(history['tl']) + 1)

for ax, (title, k1, k2, c1, c2) in zip(axes[:2], [
    ('Loss',     'tl', 'vl', '#42a5f5', '#ef5350'),
    ('Accuracy', 'ta', 'va', '#66bb6a', '#ffa726'),
]):
    ax.set_facecolor('#161b22')
    ax.plot(ep_range, history[k1], color=c1, lw=2, label='Train')
    ax.plot(ep_range, history[k2], color=c2, lw=2, linestyle='--', label='Val')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', color='#aaa'); ax.tick_params(colors='white')
    ax.legend(facecolor='#0d1117', labelcolor='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

ax3 = axes[2]; ax3.set_facecolor('#161b22')
ax3.plot(ep_range, history['auc'], color='#ab47bc', lw=2, label='Val AUC')
ax3.set_ylabel('AUC', color='#ab47bc'); ax3.tick_params(axis='y', labelcolor='#ab47bc', colors='white')
ax3t = ax3.twinx()
ax3t.plot(ep_range, history['lr'], color='#ffee58', lw=1.5, linestyle=':', label='LR')
ax3t.set_ylabel('LR', color='#ffee58'); ax3t.tick_params(axis='y', labelcolor='#ffee58')
ax3.set_title('AUC & LR', color='white', fontsize=13, fontweight='bold')
ax3.set_xlabel('Epoch', color='#aaa'); ax3.tick_params(colors='white')
for sp in ax3.spines.values(): sp.set_edgecolor('#333')

plt.tight_layout()
plt.savefig(str(DRIVE_DIR / 'training_curves.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
""".splitlines(keepends=True)))

# load best checkpoint
cells.append(code("""\
ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'Loaded checkpoint from epoch {ckpt["epoch"]}  AUC={ckpt["val_auc"]:.4f}')
""".splitlines(keepends=True)))

# test evaluation
cells.append(code("""\
tl, acc, auc, probs, labs = eval_epoch(model, test_loader, criterion, DEVICE)
preds = (probs >= 0.5).astype(int)
print(f'Test Accuracy : {acc*100:.2f}%')
print(f'Test AUC-ROC  : {auc:.4f}')
print()
print(classification_report(labs, preds, target_names=['Real', 'Fake']))
""".splitlines(keepends=True)))

# confusion matrix + ROC
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0d1117')

cm = confusion_matrix(labs, preds)
ax = axes[0]; ax.set_facecolor('#161b22')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_title('Confusion Matrix', color='white', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', color='#aaa'); ax.set_ylabel('Actual', color='#aaa')
ax.tick_params(colors='white')

fpr, tpr, _ = roc_curve(labs, probs)
ax2 = axes[1]; ax2.set_facecolor('#161b22')
ax2.plot(fpr, tpr, color='#42a5f5', lw=2.5, label=f'AUC = {auc:.3f}')
ax2.plot([0, 1], [0, 1], '--', color='#555', lw=1.5, label='Random')
ax2.fill_between(fpr, tpr, alpha=0.15, color='#42a5f5')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
ax2.set_xlabel('False Positive Rate', color='#aaa'); ax2.set_ylabel('True Positive Rate', color='#aaa')
ax2.set_title('ROC Curve', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white'); ax2.legend(facecolor='#0d1117', labelcolor='white', fontsize=11)
for sp in ax2.spines.values(): sp.set_edgecolor('#333')

plt.tight_layout()
plt.savefig(str(DRIVE_DIR / 'evaluation.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
""".splitlines(keepends=True)))

# GradCAM
cells.append(code("""\
target_layer = model.backbone.conv_head
cam = GradCAM(model=model, target_layers=[target_layer])
CLS  = ['Real', 'Fake']
CMAP = {'Real': '#4CAF50', 'Fake': '#F44336'}
NMEAN = np.array(MEAN)[:, None, None]
NSTD  = np.array(STD)[:, None, None]

n_show = 8
idxs = np.concatenate([np.where(labs == 0)[0][:n_show // 2], np.where(labs == 1)[0][:n_show // 2]])
fig, axes = plt.subplots(3, n_show, figsize=(22, 9))
fig.patch.set_facecolor('#0d1117')

for ci, si in enumerate(idxs):
    img_t, tl_ = test_ds[si]
    prob_ = float(probs[si]); pred_ = int(prob_ >= 0.5)
    gc = cam(input_tensor=img_t.unsqueeze(0), targets=[ClassifierOutputTarget(pred_)])[0]
    rgb = (img_t.numpy() * NSTD + NMEAN).clip(0, 1).transpose(1, 2, 0)
    rgb_r = cv2.resize(rgb.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    gc_r  = cv2.resize(gc, (IMG_SIZE, IMG_SIZE))
    overlay = show_cam_on_image(rgb_r, gc_r, use_rgb=True)
    ok = (pred_ == tl_); bc = '#4CAF50' if ok else '#F44336'

    for ri, (im, rl) in enumerate([(rgb_r, 'Original'), (overlay, 'GradCAM'), (None, 'Pred')]):
        ax = axes[ri, ci]; ax.set_facecolor('#161b22')
        if ri < 2:
            ax.imshow(im)
            for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(2.5); sp.set_visible(True)
        else:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.text(.5, .62, CLS[pred_], ha='center', va='center', fontsize=11, fontweight='bold', color=CMAP[CLS[pred_]])
            s = 'correct' if ok else 'wrong'
            ax.text(.5, .3, f'{s} (GT:{CLS[tl_]})', ha='center', va='center', fontsize=8,
                    color='white' if ok else '#ff6b6b')
            for sp in ax.spines.values(): sp.set_edgecolor('#333'); sp.set_visible(True)
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel(rl, color='white', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(str(DRIVE_DIR / 'gradcam.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
""".splitlines(keepends=True)))

# predict function
cells.append(code("""\
@torch.no_grad()
def predict_image(src):
    model.eval()
    img = Image.open(src).convert('RGB') if isinstance(src, (str, Path)) else src.convert('RGB')
    t = val_tfm(img).to(DEVICE)
    prob = float(torch.softmax(model(t.unsqueeze(0)), 1)[0, 1])
    lbl  = 'Fake' if prob >= 0.5 else 'Real'
    conf = prob if lbl == 'Fake' else 1 - prob
    return {'label': lbl, 'confidence': conf, 'fake_prob': prob, 'real_prob': 1 - prob}

def show_prediction(src):
    res = predict_image(src)
    img = Image.open(src).convert('RGB') if isinstance(src, (str, Path)) else src.convert('RGB')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    ax1 = axes[0]; ax1.imshow(img.resize((380, 380))); ax1.axis('off')
    ax1.set_title('Input Image', color='white', fontsize=13)
    vc = '#F44336' if res['label'] == 'Fake' else '#4CAF50'
    for sp in ax1.spines.values(): sp.set_edgecolor(vc); sp.set_linewidth(3); sp.set_visible(True)

    ax2 = axes[1]; ax2.set_facecolor('#161b22')
    bars = ax2.barh(['Real', 'Fake'], [res['real_prob'], res['fake_prob']],
                    color=['#4CAF50', '#F44336'], height=0.5, edgecolor='white', linewidth=0.5)
    for bar, p in zip(bars, [res['real_prob'], res['fake_prob']]):
        ax2.text(min(p + .01, .95), bar.get_y() + bar.get_height() / 2,
                 f'{p*100:.1f}%', va='center', color='white', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1); ax2.set_xlabel('Probability', color='#aaa')
    ax2.set_title('Confidence', color='white', fontsize=13)
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#333')

    verdict = 'DEEPFAKE DETECTED' if res['label'] == 'Fake' else 'AUTHENTIC FACE'
    fig.suptitle(f'{verdict}  ({res["confidence"]*100:.1f}%)', color=vc, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()
    return res
""".splitlines(keepends=True)))

# upload widget
cells.append(code("""\
import ipywidgets as widgets
from IPython.display import display, clear_output
import io

out = widgets.Output()
btn = widgets.FileUpload(accept='image/*', multiple=False, description='Upload Image',
                         button_style='info', layout=widgets.Layout(width='220px'))

def on_upload(change):
    with out:
        clear_output(wait=True)
        if btn.value:
            fdata = list(btn.value.values())[0]['content']
            img = Image.open(io.BytesIO(fdata))
            res = show_prediction(img)
            print(res)

btn.observe(on_upload, names='value')
display(widgets.VBox([btn, out]))
""".splitlines(keepends=True)))

# quick random test
cells.append(code("""\
idx = np.random.randint(len(test_df))
row = test_df.iloc[idx]
print('Ground truth:', 'Real' if row['label'] == 0 else 'Fake')
res = show_prediction(row['path'])
print(res)
""".splitlines(keepends=True)))

# assemble and write
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"}
    },
    "cells": cells
}

out_path = r"c:\Users\Surface\Documents\DeepGuard-AI\DeepGuard_AI.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
sz = os.path.getsize(out_path) / 1024
print(f"Done -> {out_path}")
print(f"Size: {sz:.1f} KB  |  Cells: {len(cells)}")
