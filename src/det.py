"""Hito 2: detección densa de manos/gestos reutilizando el encoder del Hito 1.

Sigue la estructura del notebook de clase "Detección de múltiples objetos con un encoder
preentrenado" (BCCD), cambiando DINOv2 por el encoder ResNet18 del Hito 1:

  imagen 288x384 -> encoder (layer3 + layer4 reescalada, stride 16) -> grilla 18x24 de 768 canales
  -> cabeza densa conv (1x1 -> 3x3 -> 1x1) -> por celda: [objectness, tx, ty, tw, th, clases...]

Entrenamiento: asignación por celda del centro, focal loss (objectness) + 1-GIoU (caja) + CE (clase).
Inferencia: score = σ(obj)·p(clase) -> umbral -> NMS por clase.
Evaluación: AP estilo Pascal VOC (todos los puntos) por clase y umbral de IoU -> mAP@0.5 y mAP@[.5:.95].

Formato de cajas: (xc, yc, w, h) normalizadas al lienzo de entrada (W=288, H=384) para el modelo,
xyxy normalizadas a la imagen ORIGINAL para guardar predicciones (sirve para recortar y analizar).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.ops import batched_nms

from src.data import CLASSES, MEAN, ROOT, STD
from src.models import ResNet18Encoder

IMG_W, IMG_H = 288, 384           # múltiplos de 16; la mayoría de las imágenes de HaGRID son 3:4 vertical
STRIDE = 16
GRID_W, GRID_H = IMG_W // STRIDE, IMG_H // STRIDE   # 18 x 24 celdas
FEAT_DIM = 256 + 512              # layer3 + layer4 de ResNet18
PAD_VALUE = 114                   # gris del relleno (letterbox)


def class_names(mode: str):
    return ["hand"] if mode == "hands" else CLASSES


# ----------------------------------------------------------------------------------------------
# Cajas e IoU (mismas funciones del notebook de clase)
# ----------------------------------------------------------------------------------------------
def bbox_xywh_to_xyxy(b):
    xc, yc, w, h = b.unbind(-1)
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1)


def giou_per_box(pred, target, eps=1e-7):
    """GIoU elemento a elemento entre [N, 4] y [N, 4] en formato (xc, yc, w, h)."""
    p, t = bbox_xywh_to_xyxy(pred), bbox_xywh_to_xyxy(target)
    ix1, iy1 = torch.max(p[..., 0], t[..., 0]), torch.max(p[..., 1], t[..., 1])
    ix2, iy2 = torch.min(p[..., 2], t[..., 2]), torch.min(p[..., 3], t[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
    area_t = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    cx1, cy1 = torch.min(p[..., 0], t[..., 0]), torch.min(p[..., 1], t[..., 1])
    cx2, cy2 = torch.max(p[..., 2], t[..., 2]), torch.max(p[..., 3], t[..., 3])
    area_c = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    return iou - (area_c - union) / area_c


def np_iou_matrix(a, b, eps=1e-9):
    """IoU por pares entre a [N, 4] y b [M, 4] (xyxy)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
def letterbox_params(width, height):
    """Escala y desplazamiento para encajar la imagen en el lienzo IMG_W x IMG_H sin deformarla."""
    scale = min(IMG_W / width, IMG_H / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (IMG_W - new_w) // 2, (IMG_H - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def to_canvas(boxes_xywh_tl, width, height):
    """Cajas HaGRID [x, y, w, h] (esquina sup. izq., normalizadas a la imagen) -> (xc, yc, w, h) del lienzo."""
    scale, _, _, pad_x, pad_y = letterbox_params(width, height)
    b = np.asarray(boxes_xywh_tl, dtype=np.float32).reshape(-1, 4)
    x = (b[:, 0] * width * scale + pad_x) / IMG_W
    y = (b[:, 1] * height * scale + pad_y) / IMG_H
    w = b[:, 2] * width * scale / IMG_W
    h = b[:, 3] * height * scale / IMG_H
    return np.stack([x + w / 2, y + h / 2, w, h], axis=1)


def canvas_xyxy_to_original(boxes_xyxy, width, height):
    """Cajas xyxy normalizadas al lienzo -> xyxy normalizadas a la imagen original (recortadas a [0, 1])."""
    scale, _, _, pad_x, pad_y = letterbox_params(width, height)
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] * IMG_W - pad_x) / scale / width
    b[:, [1, 3]] = (b[:, [1, 3]] * IMG_H - pad_y) / scale / height
    return np.clip(b, 0, 1)


class HagridDetDataset(Dataset):
    """Imagen completa en letterbox 288x384 + todas sus manos.

    mode="gestures": 11 clases (10 gestos + no_gesture). mode="hands": 1 clase (toda mano).
    Devuelve (imagen normalizada [3, H, W], cajas [K, 5] = (clase, xc, yc, w, h) del lienzo, índice).
    """

    def __init__(self, split: str, mode: str = "gestures", train: bool = False, limit: int | None = None):
        self.items = json.load(open(ROOT / f"data/splits/det_{split}.json"))
        if limit:
            self.items = self.items[:limit]
        self.names = class_names(mode)
        self.mode = mode
        self.train = train
        self.color = T.ColorJitter(0.3, 0.3, 0.3, 0.05)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def labels_of(self, item):
        if self.mode == "hands":
            return [0] * len(item["labels"])
        return [self.names.index(l) for l in item["labels"]]

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(ROOT / "data/images" / it["folder"] / f"{it['image_id']}.jpg").convert("RGB")
        _, new_w, new_h, pad_x, pad_y = letterbox_params(*img.size)
        canvas = Image.new("RGB", (IMG_W, IMG_H), (PAD_VALUE,) * 3)
        canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
        boxes = to_canvas(it["boxes"], it["width"], it["height"])
        if self.train:
            canvas = self.color(canvas)
            if random.random() < 0.5:   # volteo horizontal: ningún gesto depende de la mano izq./der.
                canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = 1 - boxes[:, 0]
        target = np.concatenate([np.array(self.labels_of(it), dtype=np.float32)[:, None], boxes], axis=1)
        return self.norm(canvas), torch.from_numpy(target), i


def collate(batch):
    imgs, targets, idx = zip(*batch)
    k = max(len(t) for t in targets)
    padded = torch.full((len(targets), k, 5), -1.0)
    for j, t in enumerate(targets):
        padded[j, : len(t)] = t
    return torch.stack(imgs), padded, torch.tensor(idx)


# ----------------------------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------------------------
class ResNet18Features(nn.Module):
    """Encoder ResNet18 (del Hito 1 o ImageNet) -> mapa de features [B, 768, 24, 18] con stride 16."""

    def __init__(self, init: str = "hito1"):
        super().__init__()
        enc = ResNet18Encoder(pretrained=(init == "imagenet"))
        if init == "hito1":
            state = torch.load(ROOT / "checkpoints/resnet18_finetune/encoder.pt", map_location="cpu")["encoder"]
            enc.load_state_dict(state)
        n = enc.net
        self.stem = nn.Sequential(n.conv1, n.bn1, n.relu, n.maxpool, n.layer1, n.layer2)
        self.layer3, self.layer4 = n.layer3, n.layer4

    def forward(self, x):
        c3 = self.layer3(self.stem(x))                 # stride 16, 256 canales
        c4 = self.layer4(c3)                           # stride 32, 512 canales
        c4 = F.interpolate(c4, size=c3.shape[-2:], mode="nearest")
        return torch.cat([c3, c4], dim=1)


class DenseDetectionHeadConv(nn.Module):
    """Cabeza del notebook de clase: Conv 1x1 -> GELU -> Conv 3x3 -> GELU -> Conv 1x1."""

    def __init__(self, in_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 5 + num_classes, 1),
        )
        prior = 0.01                                   # P(objeto) inicial ~0.01
        with torch.no_grad():
            self.net[-1].bias[0] = -float(np.log((1 - prior) / prior))

    def forward(self, feats):
        return self.net(feats).permute(0, 2, 3, 1).contiguous()   # [B, Hg, Wg, 5 + C]


class HandDetector(nn.Module):
    def __init__(self, num_classes, init="hito1", train_backbone="none"):
        super().__init__()
        self.backbone = ResNet18Features(init)
        self.head = DenseDetectionHeadConv(FEAT_DIM, num_classes)
        self.train_backbone = train_backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        if train_backbone == "layer34":
            for p in list(self.backbone.layer3.parameters()) + list(self.backbone.layer4.parameters()):
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.train_backbone == "none":
            self.backbone.eval()                       # encoder congelado: BatchNorm en modo inferencia
        else:
            self.backbone.stem.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x))


def build_detector(cfg):
    return HandDetector(len(class_names(cfg["mode"])), cfg["init"], cfg.get("train_backbone", "none"))


def load_detector(ckpt_path, device="cpu"):
    # weights_only=False: checkpoints propios (los primeros guardaban métricas como escalares numpy)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = HandDetector(len(class_names(cfg["mode"])), "none", cfg.get("train_backbone", "none"))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


# ----------------------------------------------------------------------------------------------
# Asignación y pérdida (notebook de clase, secciones 7 y 8)
# ----------------------------------------------------------------------------------------------
def encode_targets(boxes, grid_h=GRID_H, grid_w=GRID_W):
    """boxes [B, K, 5] (clase, xc, yc, w, h) con padding clase=-1 -> (obj_t, cls_t, box_t).
    El objeto se asigna a la celda que contiene su centro; en colisión gana el de mayor área."""
    B, K, _ = boxes.shape
    obj_t = torch.zeros(B, grid_h, grid_w)
    cls_t = torch.full((B, grid_h, grid_w), -1, dtype=torch.long)
    box_t = torch.zeros(B, grid_h, grid_w, 4)
    for b in range(B):
        for c, xc, yc, w, h in boxes[b].tolist():
            if c < 0:
                continue
            gx, gy = min(int(xc * grid_w), grid_w - 1), min(int(yc * grid_h), grid_h - 1)
            if obj_t[b, gy, gx] == 1 and w * h <= box_t[b, gy, gx, 2] * box_t[b, gy, gx, 3]:
                continue
            obj_t[b, gy, gx] = 1.0
            cls_t[b, gy, gx] = int(c)
            box_t[b, gy, gx] = torch.tensor([xc, yc, w, h])
    return obj_t, cls_t, box_t


def cells_to_boxes(raw_box):
    """(tx, ty, tw, th) por celda -> (xc, yc, w, h) normalizadas."""
    _, Hg, Wg, _ = raw_box.shape
    ys = torch.arange(Hg, device=raw_box.device).view(1, Hg, 1).float()
    xs = torch.arange(Wg, device=raw_box.device).view(1, 1, Wg).float()
    xc = (xs + torch.sigmoid(raw_box[..., 0])) / Wg
    yc = (ys + torch.sigmoid(raw_box[..., 1])) / Hg
    return torch.stack([xc, yc, torch.sigmoid(raw_box[..., 2]), torch.sigmoid(raw_box[..., 3])], dim=-1)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t) ** gamma * ce


LAMBDA_OBJ, LAMBDA_BOX, LAMBDA_CLS = 1.0, 2.0, 1.0


def detection_loss(raw, targets):
    """raw [B, Hg, Wg, 5 + C]. Objectness con focal (todas las celdas, / n_pos); caja y clase solo en positivos."""
    obj_t, cls_t, box_t = targets
    pos = obj_t > 0.5
    n_pos = pos.sum().clamp(min=1).float()
    l_obj = sigmoid_focal_loss(raw[..., 0], obj_t).sum() / n_pos
    if pos.any():
        pred_box = cells_to_boxes(raw[..., 1:5])[pos]
        l_box = (1 - giou_per_box(pred_box, box_t[pos])).mean()
        l_cls = F.cross_entropy(raw[..., 5:][pos], cls_t[pos]) if raw.shape[-1] > 6 else raw.new_zeros(())
    else:
        l_box = l_cls = raw.new_zeros(())
    total = LAMBDA_OBJ * l_obj + LAMBDA_BOX * l_box + LAMBDA_CLS * l_cls
    return total, {"obj": l_obj.item(), "box": l_box.item(), "cls": l_cls.item()}


# ----------------------------------------------------------------------------------------------
# Decodificación + NMS (sección 9)
# ----------------------------------------------------------------------------------------------
def decode_predictions(raw, score_thr=0.05, nms_iou=0.5, max_det=20):
    """raw [N, Hg, Wg, 5 + C] -> lista de dicts {'boxes' xyxy del lienzo, 'scores', 'labels'} (numpy)."""
    boxes_all = bbox_xywh_to_xyxy(cells_to_boxes(raw[..., 1:5])).clamp(0, 1)
    p_obj = torch.sigmoid(raw[..., 0])
    if raw.shape[-1] > 6:
        cls_score, cls_id = torch.softmax(raw[..., 5:], dim=-1).max(dim=-1)
    else:
        cls_score, cls_id = torch.ones_like(p_obj), torch.zeros_like(p_obj, dtype=torch.long)
    score_all = p_obj * cls_score
    out = []
    for n in range(raw.shape[0]):
        s, l, b = score_all[n].reshape(-1), cls_id[n].reshape(-1), boxes_all[n].reshape(-1, 4)
        m = s >= score_thr
        s, l, b = s[m], l[m], b[m]
        keep = batched_nms(b, s, l, nms_iou)[:max_det]
        out.append({"boxes": b[keep].numpy(), "scores": s[keep].numpy(), "labels": l[keep].numpy()})
    return out


# ----------------------------------------------------------------------------------------------
# Métricas (sección 10): AP Pascal VOC por clase y umbral de IoU
# ----------------------------------------------------------------------------------------------
def average_precision(tp, scores, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_image(pred, gt_boxes, gt_labels, iou_thr, num_classes):
    """Matching greedy por clase en UNA imagen. Devuelve {c: (scores, tp, n_gt, ious_tp)}."""
    res = {}
    for c in range(num_classes):
        pm, gm = pred["labels"] == c, gt_labels == c
        pb, ps, gb = pred["boxes"][pm], pred["scores"][pm], gt_boxes[gm]
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        ious = np_iou_matrix(pb, gb)
        used = np.zeros(len(gb), bool)
        tp = np.zeros(len(pb), bool)
        tp_iou = []
        for i in range(len(pb)):
            if len(gb) == 0:
                break
            cand = np.where(~used, ious[i], -1)
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                used[j] = tp[i] = True
                tp_iou.append(cand[j])
        res[c] = (ps, tp, int(gm.sum()), tp_iou)
    return res


def evaluate_detections(preds, gts, names, op_score=0.5, iou_thrs=np.arange(0.5, 0.96, 0.05)):
    """preds: lista de dicts (xyxy); gts: lista de (boxes xyxy [n,4], labels [n]).
    Retorna dict con mAP50, mAP50_95, AP50 por clase, precision/recall/F1 en op_score (IoU 0.5) e IoU medio de TP."""
    C = len(names)
    ap = np.zeros((len(iou_thrs), C))
    out = {}
    for t_i, thr in enumerate(iou_thrs):
        acc = {c: ([], [], 0, []) for c in range(C)}
        for p, (gb, gl) in zip(preds, gts):
            for c, (s, tp, n, ti) in match_image(p, gb, gl, thr, C).items():
                a = acc[c]
                acc[c] = (a[0] + [s], a[1] + [tp], a[2] + n, a[3] + ti)
        for c in range(C):
            s = np.concatenate(acc[c][0]) if acc[c][0] else np.zeros(0)
            tp = np.concatenate(acc[c][1]) if acc[c][1] else np.zeros(0, bool)
            ap[t_i, c] = average_precision(tp, s, acc[c][2])
            if abs(thr - 0.5) < 1e-9:
                out[f"AP50_{names[c]}"] = ap[t_i, c]
                if c == 0:
                    tot_tp = tot_fp = tot_gt = 0
                    all_ious = []
                sel = s >= op_score
                tot_tp += int(tp[sel].sum())
                tot_fp += int((~tp[sel]).sum())
                tot_gt += acc[c][2]
                all_ious += acc[c][3]
    out["mAP50"] = float(np.nanmean(ap[0]))
    out["mAP50_95"] = float(np.nanmean(ap))
    out["precision"] = tot_tp / max(tot_tp + tot_fp, 1)
    out["recall"] = tot_tp / max(tot_gt, 1)
    out["F1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-9)
    out["mean_IoU_TP"] = float(np.mean(all_ious)) if all_ious else 0.0
    return out


def gts_from_dataset(ds):
    """GT de cada imagen del dataset en xyxy del lienzo (sin aumentos)."""
    gts = []
    for it in ds.items:
        b = torch.from_numpy(to_canvas(it["boxes"], it["width"], it["height"]))
        gts.append((bbox_xywh_to_xyxy(b).numpy(), np.array(ds.labels_of(it))))
    return gts


@torch.no_grad()
def predict(model, loader, device, score_thr=0.05):
    model.eval()
    preds = []
    for x, _, _ in loader:
        preds += decode_predictions(model(x.to(device)).float().cpu(), score_thr=score_thr)
    return preds
