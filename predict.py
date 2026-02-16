import os
import cv2
import torch
import argparse
import numpy as np
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from transformers import DistilBertTokenizerFast, DistilBertModel

# ======== CONSTANTS (must match training) ========
NUM_FRAMES = 16
MAX_AUDIO_FRAMES = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== TOKENIZER ========
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)


# ================== VIDEO TRANSFORM ==================
video_transform_none = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================== FACE EXTRACTION ==================
def extract_faces(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames

    frame_idxs = np.linspace(0, total - 1, num_frames).astype(int)
    idx_set = set(frame_idxs.tolist())

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frames.append(cv2.resize(frame, (224, 224)))
        i += 1

    cap.release()
    return frames


# ======== MODEL DEFINITION ========
class MultiModalNetNoAnn(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.lstm = nn.LSTM(512, hidden, batch_first=True, bidirectional=True)
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.audio_lstm = nn.LSTM(40, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden*2 + 768 + 128*2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, faces, tokens, mfcc):
        B,T,C,H,W = faces.shape
        faces = faces.view(B*T, C, H, W)

        feat = self.cnn(faces)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        vid_feat = out.mean(dim=1)

        bert_feat = self.bert(**tokens).last_hidden_state[:,0,:]

        aud_out, _ = self.audio_lstm(mfcc)
        aud_feat = aud_out.mean(dim=1)

        fused = torch.cat([vid_feat, bert_feat, aud_feat], dim=1)
        return self.fc(fused).squeeze(1)

# ======== PREPROCESS VIDEO ========
def preprocess_video(video_path):
    faces = extract_faces(video_path, NUM_FRAMES)
    if len(faces) == 0:
        blank = np.zeros((224,224,3), dtype=np.uint8) + 127
        faces = [blank] * NUM_FRAMES

    frames = []
    for f in faces[:NUM_FRAMES]:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(video_transform_none(img))

    while len(frames) < NUM_FRAMES:
        frames.append(torch.zeros_like(frames[0]))

    return torch.stack(frames).unsqueeze(0)

# ======== PREPROCESS TEXT ========
def preprocess_text(video_path, txt_dir):
    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    txt_file = os.path.join(txt_dir, vid_id + ".txt")

    text = ""
    if os.path.exists(txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return {k: v.to(DEVICE) for k, v in tokens.items()}

# ======== PREPROCESS AUDIO ========
def preprocess_audio(video_path, audio_dir=None):
    """
    Robust audio preprocessing:
    - Try loading audio
    - If it fails OR no audio → return zero MFCCs
    """

    try:
        # This will FAIL for MP4 on Windows → expected
        wav, sr = torchaudio.load(video_path)

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 64
            }
        )(wav)

        mfcc = mfcc.squeeze(0).transpose(0, 1)

    except Exception:
        # 🔥 FALLBACK: NO AUDIO → ZERO MFCC
        mfcc = torch.zeros((MAX_AUDIO_FRAMES, 40))

    # Pad / truncate
    if mfcc.shape[0] > MAX_AUDIO_FRAMES:
        mfcc = mfcc[:MAX_AUDIO_FRAMES]
    else:
        mfcc = F.pad(
            mfcc,
            (0, 0, 0, MAX_AUDIO_FRAMES - mfcc.shape[0])
        )

    return mfcc.unsqueeze(0)

# ======== MAIN PREDICTION ========
def main(args):
    model = MultiModalNetNoAnn()
    model.load_state_dict(
        torch.load(args.model, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    faces = preprocess_video(args.video).to(DEVICE)
    tokens = preprocess_text(args.video, args.txt_dir)
    mfcc = preprocess_audio(args.video, args.audio_dir).to(DEVICE)

    with torch.no_grad():
        prob = model(faces, tokens, mfcc).item()

    label = "LIE ❌" if prob > 0.8 else "TRUTH ✅"
    confidence = prob if prob > 0.5 else (1 - prob)

    print("\n=== DECEPTION PREDICTION ===")
    print(f"Result     : {label}")
    print(f"Confidence : {confidence*100:.2f}%")

# ======== ENTRY ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
    "--model",
    default=r"D:\model and file\deception_model_noann.pth"
)
    parser.add_argument("--txt_dir", default="transcript")
    parser.add_argument("--audio_dir", default="audio_cache")
    args = parser.parse_args()

    main(args)
