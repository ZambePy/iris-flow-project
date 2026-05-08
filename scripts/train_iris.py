"""
IrisFlow — scripts/train_iris.py

Pipeline de treino da IrisGazeNet em duas fases:
  Fase 1 — Pre-treino com OpenEDS (50 epocas):
      Input:  frame completo do headset VR (400x640 gray, valores 0-3) -> resize 64x128 RGB
      Target: centro normalizado do bbox da iris [cx/W, cy/H]
      Aprende a localizar a iris antes de ver dados de tela.

  Fase 2 — Treino principal com MPIIGaze (100 epocas):
      Input:  imagem do olho (36x60 gray) -> resize 224x224 RGB + head_pose (N, 3)
      Target: vetor de gaze (yaw, pitch) em radianos convertido para [0, 1]
      Mapeia olhar -> posicao na tela.

Resume automatico:
  Se iris_gaze_checkpoint_fase*_ep*.pth ou iris_gaze_model.best.pth existirem,
  o treino continua de onde parou. Checkpoints completos (pesos + optimizer +
  scheduler + epoca) sao salvos a cada 10 epocas e sempre que o val_loss melhora.

Rodar com:
    .venv\\Scripts\\python.exe scripts/train_iris.py

Saida:
    engine/iris_gaze_model.pth
    logs/train_iris.log
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from iris_model import IrisGazeNet  # noqa: E402

# ── Caminhos ──────────────────────────────────────────────────────────────────

OPENEDS_IMG_DIR  = ROOT / "datasets" / "OpenEDS" / "openEDS" / "openEDS"
OPENEDS_BBOX_DIR = ROOT / "datasets" / "OpenEDS" / "bbox" / "bbox"
MPIIGAZE_NORM    = ROOT / "datasets" / "MPIIGaze" / "archive" / "MPIIGaze" / "Data" / "Normalized"
MODEL_OUT        = ROOT / "engine" / "iris_gaze_model.pth"
BEST_PATH        = MODEL_OUT.with_suffix(".best.pth")
LOG_DIR          = ROOT / "logs"

# ── Hiperparametros ───────────────────────────────────────────────────────────

BATCH_SIZE    = 16
LR_PHASE1     = 1e-4   # MobileNetV2 pré-treinado é sensível a LR alta
LR_PHASE2     = 1e-5
PHASE1_EPOCHS = 50
PHASE2_EPOCHS = 100
EYE_H, EYE_W  = 224, 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHASE_ORDER = ["Fase1-OpenEDS", "Fase2-MPIIGaze"]

# Head pose do MPIIGaze: normalizar radianos para [-1, 1]
_POSE_SCALE = np.array([np.pi / 2, np.pi / 2, np.pi / 6], dtype=np.float32)

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train_iris.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("train_iris")


# ── Utilitarios de imagem ─────────────────────────────────────────────────────

def _gray_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Converte array grayscale (H, W) para uint8 RGB (H, W, 3).
    Trata o caso OpenEDS onde os valores estao em [0, 3]: escala para [0, 255].
    """
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    vmax = int(arr.max())
    if vmax < 16:
        arr = (arr.astype(np.float32) / max(vmax, 1) * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _resize_eye(img: np.ndarray) -> np.ndarray:
    """Resize (H, W, 3) para (EYE_H, EYE_W, 3)."""
    return cv2.resize(img, (EYE_W, EYE_H))


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> tensor (3, H, W) float32 em [0, 1]."""
    return TF.to_tensor(img).float()


# ── Augmentacao ───────────────────────────────────────────────────────────────

def _augment(
    img: np.ndarray, label: np.ndarray, flip: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    img:   (H, W, 3) uint8
    label: (2,) float32 em [0, 1]
    Retorna (tensor CHW float32, label augmentado).
    """
    from PIL import Image
    pil = Image.fromarray(img)

    import torchvision.transforms as T
    pil = T.ColorJitter(brightness=0.3, contrast=0.3)(pil)

    if np.random.rand() < 0.2:
        pil = T.GaussianBlur(kernel_size=3)(pil)

    angle = float(np.random.uniform(-5, 5))
    pil   = TF.rotate(pil, angle)

    if flip and np.random.rand() < 0.3:
        pil      = TF.hflip(pil)
        label    = label.copy()
        label[0] = 1.0 - label[0]

    tensor = TF.to_tensor(pil).float()
    tensor = (tensor + torch.randn_like(tensor) * 0.02).clamp(0.0, 1.0)
    return tensor, label


# ── Dataset OpenEDS ───────────────────────────────────────────────────────────

class OpenEDSDataset(Dataset):
    """
    Pre-treino: localizar iris dentro de frames completos do headset VR.

    Imagens: (400, 640) grayscale uint8, valores em [0, 3].
    Bbox:    x_min x_max y_min y_max — marca a regiao da iris.
    Target:  centro normalizado do bbox [cx/W, cy/H] em [0, 1].
    """

    def __init__(self, augment: bool = True) -> None:
        self.augment = augment
        self.images: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self._load()

    def _load(self) -> None:
        if not OPENEDS_IMG_DIR.exists():
            log.warning("OpenEDS nao encontrado em %s", OPENEDS_IMG_DIR)
            return

        subj_dirs = [d for d in sorted(OPENEDS_IMG_DIR.iterdir()) if d.is_dir()]
        log.info("OpenEDS: carregando %d sujeitos em RAM...", len(subj_dirs))

        for subj_dir in subj_dirs:
            subj_id  = subj_dir.name
            bbox_txt = OPENEDS_BBOX_DIR / f"{subj_id}.txt"
            if not bbox_txt.exists():
                continue

            with open(bbox_txt) as f:
                lines = f.read().splitlines()

            img_files = sorted(
                (p for p in subj_dir.glob("*.npy") if p.stem.isdigit()),
                key=lambda p: int(p.stem),
            )
            if not img_files:
                img_files = sorted(
                    (p for p in subj_dir.glob("*.png") if p.stem.isdigit()),
                    key=lambda p: int(p.stem),
                )

            for i, img_path in enumerate(img_files):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                if len(parts) < 4:
                    continue
                try:
                    x_min, x_max, y_min, y_max = (int(v) for v in parts[:4])
                except ValueError:
                    continue
                if x_max <= x_min or y_max <= y_min:
                    continue

                try:
                    if img_path.suffix == ".npy":
                        raw = np.load(str(img_path))
                    else:
                        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if raw is None:
                            continue
                except Exception:
                    continue

                img = _gray_to_rgb_uint8(raw)
                img = _resize_eye(img)

                cx = (x_min + x_max) / 2.0 / 640.0
                cy = (y_min + y_max) / 2.0 / 400.0
                label = np.array([
                    float(np.clip(cx, 0.0, 1.0)),
                    float(np.clip(cy, 0.0, 1.0)),
                ], dtype=np.float32)

                self.images.append(img)
                self.labels.append(label)

        mb = len(self.images) * EYE_H * EYE_W * 3 / 1024 / 1024
        log.info("OpenEDS: %d amostras em RAM (%.0f MB).", len(self.images), mb)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img   = self.images[idx]
        label = self.labels[idx].copy()

        if self.augment:
            tensor, label = _augment(img, label, flip=True)
        else:
            tensor = _to_tensor(img)

        head_pose = torch.zeros(3, dtype=torch.float32)
        return tensor, head_pose, torch.from_numpy(label)


# ── Dataset MPIIGaze ──────────────────────────────────────────────────────────

class MPIIGazeDataset(Dataset):
    """
    Treino principal: mapear imagem do olho + head pose -> posicao na tela.

    Imagens: (N, 36, 60) grayscale uint8 (campo 'image' do .mat normalizado).
    Head pose: (N, 3) float64 em radianos (campo 'pose'), normalizado para [-1, 1].
    Target: vetor de gaze (yaw, pitch) em radianos convertido para [0, 1].
            yaw  em [-0.6, 0.6] rad → [0, 1] horizontal
            pitch em [-0.6, 0.6] rad → [0, 1] vertical
    """

    def __init__(self, augment: bool = True) -> None:
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._load()

    def _load(self) -> None:
        if not MPIIGAZE_NORM.exists():
            log.warning("MPIIGaze nao encontrado em %s", MPIIGAZE_NORM)
            return

        try:
            import scipy.io
        except ImportError:
            log.error("scipy necessario: pip install scipy")
            return

        mat_files = sorted(MPIIGAZE_NORM.rglob("*.mat"))
        log.info("MPIIGaze: %d arquivos .mat encontrados.", len(mat_files))

        for mat_path in mat_files:
            try:
                mat   = scipy.io.loadmat(str(mat_path))
                left  = mat["data"][0, 0]["left"][0, 0]
                imgs  = left["image"]   # (N, 36, 60) grayscale
                poses = left["pose"]    # (N, 3) radianos
                gazes = left["gaze"]    # (N, 2) radianos — yaw, pitch do olhar
            except Exception as e:
                log.debug("Falha ao carregar %s: %s", mat_path.name, e)
                continue

            for i in range(len(imgs)):
                img_array = imgs[i].astype(np.float32)
                pose      = poses[i].astype(np.float32)
                gaze_x    = float((gazes[i, 0] + 0.6) / 1.2)
                gaze_y    = float((gazes[i, 1] + 0.6) / 1.2)
                label     = np.array([gaze_x, gaze_y], dtype=np.float32)

                if not (np.isfinite(img_array).all() and np.isfinite(pose).all() and np.isfinite(label).all()):
                    continue

                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                img_rgb   = _gray_to_rgb_uint8(img_array)
                pose_norm = np.clip(pose / _POSE_SCALE, -1.0, 1.0)
                label     = np.array([
                    float(np.clip(gaze_x, 0.0, 1.0)),
                    float(np.clip(gaze_y, 0.0, 1.0)),
                ], dtype=np.float32)

                self.samples.append((img_rgb, pose_norm, label))

        log.info("MPIIGaze: %d amostras carregadas.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_rgb, pose, label = self.samples[idx]

        img = _resize_eye(img_rgb)

        if self.augment:
            tensor, label = _augment(img, label, flip=True)
        else:
            tensor = _to_tensor(img)

        return tensor, torch.from_numpy(pose), torch.from_numpy(label)


# ── Loop de treino ────────────────────────────────────────────────────────────

def _loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return 0.7 * nn.functional.mse_loss(preds, labels) \
         + 0.3 * nn.functional.l1_loss(preds, labels)


def _train_epoch(
    model: IrisGazeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total = 0.0
    for batch_idx, (imgs, poses, labels) in enumerate(loader):
        imgs, poses, labels = imgs.to(DEVICE), poses.to(DEVICE), labels.to(DEVICE)
        if not (torch.isfinite(imgs).all() and torch.isfinite(poses).all()):
            log.warning("WARNING: NaN/Inf nos inputs em batch %d — pulando", batch_idx)
            continue
        optimizer.zero_grad()
        loss = _loss(model(imgs, poses), labels)
        if not torch.isfinite(loss):
            log.warning("WARNING: NaN/Inf loss em batch %d — pulando", batch_idx)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(imgs)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def _val_epoch(model: IrisGazeNet, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss, total_px = 0.0, 0.0
    for imgs, poses, labels in loader:
        imgs, poses, labels = imgs.to(DEVICE), poses.to(DEVICE), labels.to(DEVICE)
        preds      = model(imgs, poses)
        total_loss += _loss(preds, labels).item() * len(imgs)
        err_px     = torch.sqrt(
            ((preds[:, 0] - labels[:, 0]) * 1280) ** 2
            + ((preds[:, 1] - labels[:, 1]) * 800) ** 2
        ).mean().item()
        total_px += err_px * len(imgs)
    n = max(len(loader.dataset), 1)
    return total_loss / n, total_px / n


# ── Checkpoints ───────────────────────────────────────────────────────────────

def _ckpt_path(phase: str, epoch: int) -> Path:
    tag = "fase1" if "OpenEDS" in phase else "fase2"
    return MODEL_OUT.parent / f"iris_gaze_checkpoint_{tag}_ep{epoch:03d}.pth"


def _save_full_checkpoint(
    path: Path,
    model: IrisGazeNet,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    phase: str,
    epoch: int,
    best_val: float,
) -> None:
    torch.save({
        "phase":           phase,
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val":        best_val,
    }, path)


def _find_latest_checkpoint() -> Optional[dict]:
    """
    Procura o checkpoint mais recente entre os checkpoints periodicos e o best.
    Retorna dict com chaves: phase, epoch, model_state, optimizer_state,
    scheduler_state, best_val (e opcionalmente legacy=True).
    Retorna None se nenhum checkpoint existir.
    """
    candidates: List[dict] = []

    for p in sorted(MODEL_OUT.parent.glob("iris_gaze_checkpoint_*.pth")):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "phase" in ckpt and "epoch" in ckpt:
                candidates.append(ckpt)
                log.info("Checkpoint periódico encontrado: %s  fase=%s  ep=%d",
                         p.name, ckpt["phase"], ckpt["epoch"])
        except Exception as exc:
            log.warning("Falha ao carregar checkpoint %s: %s", p.name, exc)

    if BEST_PATH.exists():
        try:
            ckpt = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "phase" in ckpt and "epoch" in ckpt:
                candidates.append(ckpt)
                log.info("Best checkpoint encontrado: fase=%s  ep=%d",
                         ckpt["phase"], ckpt["epoch"])
            else:
                # Formato legado (apenas state_dict): best.pth existe → Fase 1 concluiu
                # e o crash ocorreu durante a Fase 2.
                log.info(
                    "iris_gaze_model.best.pth detectado (formato legado, apenas pesos). "
                    "Carregando pesos e retomando Fase 2 do início."
                )
                return {
                    "phase":           "Fase2-MPIIGaze",
                    "epoch":           0,
                    "model_state":     ckpt,
                    "optimizer_state": None,
                    "scheduler_state": None,
                    "best_val":        float("inf"),
                    "legacy":          True,
                }
        except Exception as exc:
            log.warning("Falha ao carregar best checkpoint: %s", exc)

    if not candidates:
        return None

    def _rank(c: dict) -> tuple:
        pi = PHASE_ORDER.index(c["phase"]) if c["phase"] in PHASE_ORDER else -1
        return (pi, c["epoch"])

    candidates.sort(key=_rank)
    latest = candidates[-1]
    log.info("Retomando de: fase=%s  ep=%d  best_val=%.4f",
             latest["phase"], latest["epoch"], latest.get("best_val", float("inf")))
    return latest


# ── Fase de treino ────────────────────────────────────────────────────────────

def _run_phase(
    model: IrisGazeNet,
    dataset: Dataset,
    n_epochs: int,
    phase_name: str,
    lr: float = LR_PHASE1,
    resume: Optional[dict] = None,
) -> None:
    if len(dataset) == 0:
        log.warning("%s: dataset vazio — fase ignorada.", phase_name)
        return

    n_val = max(int(len(dataset) * 0.1), 1)
    n_trn = len(dataset) - n_val
    trn_ds, val_ds = random_split(
        dataset, [n_trn, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    trn_loader = DataLoader(trn_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    start_epoch = 1
    best_val    = float("inf")

    if resume is not None and resume.get("phase") == phase_name:
        start_epoch = resume["epoch"] + 1
        best_val    = resume.get("best_val", float("inf"))
        if not resume.get("legacy"):
            if resume.get("optimizer_state"):
                optimizer.load_state_dict(resume["optimizer_state"])
            if resume.get("scheduler_state"):
                scheduler.load_state_dict(resume["scheduler_state"])
        log.info("%s: retomando da época %d/%d  (best_val=%.4f)",
                 phase_name, start_epoch, n_epochs, best_val)

    if start_epoch > n_epochs:
        log.info("%s: já concluída (ep %d > %d) — pulando.", phase_name, start_epoch, n_epochs)
        return

    log.info("%s: %d treino | %d val | épocas %d→%d",
             phase_name, n_trn, n_val, start_epoch, n_epochs)

    for epoch in range(start_epoch, n_epochs + 1):
        if "MPIIGaze" in phase_name:
            frozen = epoch <= 20
            for param in model.backbone.parameters():
                param.requires_grad = not frozen
            if epoch == max(start_epoch, 1) and frozen:
                log.info("%s: backbone congelado (épocas 1–20)", phase_name)
            elif epoch == 21:
                log.info("%s: backbone descongelado na época 21", phase_name)

        try:
            trn_loss         = _train_epoch(model, trn_loader, optimizer)
            val_loss, val_px = _val_epoch(model, val_loader)
        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM na época %d — tente reduzir BATCH_SIZE.", epoch)
            torch.cuda.empty_cache()
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            _save_full_checkpoint(BEST_PATH, model, optimizer, scheduler,
                                  phase_name, epoch, best_val)

        if epoch % 10 == 0:
            ckpt_path = _ckpt_path(phase_name, epoch)
            _save_full_checkpoint(ckpt_path, model, optimizer, scheduler,
                                  phase_name, epoch, best_val)
            log.info("Checkpoint salvo: %s", ckpt_path.name)

        log.info(
            "%s  ep %03d/%d  trn=%.4f  val=%.4f  err_px=%.1f",
            phase_name, epoch, n_epochs, trn_loss, val_loss, val_px,
        )

    if BEST_PATH.exists():
        ckpt  = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        BEST_PATH.unlink(missing_ok=True)

    log.info("%s concluída. Melhor val_loss=%.4f", phase_name, best_val)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    MODEL_OUT.parent.mkdir(exist_ok=True)
    log.info("Dispositivo: %s", DEVICE)

    model  = IrisGazeNet().to(DEVICE)
    resume = _find_latest_checkpoint()

    skip_phase1 = False
    resume_p1: Optional[dict] = None
    resume_p2: Optional[dict] = None

    if resume is not None:
        model.load_state_dict(resume["model_state"])
        log.info("Pesos carregados do checkpoint (fase=%s  ep=%d).",
                 resume["phase"], resume["epoch"])

        if resume["phase"] == "Fase1-OpenEDS":
            resume_p1 = resume
        elif resume["phase"] == "Fase2-MPIIGaze":
            skip_phase1 = True
            resume_p2   = resume

    if not skip_phase1:
        openeds = OpenEDSDataset(augment=True)
        _run_phase(model, openeds, PHASE1_EPOCHS, "Fase1-OpenEDS", lr=LR_PHASE1, resume=resume_p1)
    else:
        log.info("Fase 1 (OpenEDS) ignorada — checkpoint de Fase 2 detectado.")

    mpii = MPIIGazeDataset(augment=True)
    _run_phase(model, mpii, PHASE2_EPOCHS, "Fase2-MPIIGaze", lr=LR_PHASE2, resume=resume_p2)

    torch.save(model.state_dict(), MODEL_OUT)
    log.info("Modelo salvo em %s", MODEL_OUT)
    log.info("Execute o sistema e faca calibracao para usar o modelo treinado.")


if __name__ == "__main__":
    main()
