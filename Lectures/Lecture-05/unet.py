#!/usr/bin/env python3
"""
Train a U-Net for binary segmentation of a grayscale EM image from a binary mask.

Designed for the case:
- input image: grayscale
- training mask: black/white binary mask
- limited data: patch extraction + augmentation

Example:
    python train_unet_em.py \
        --image /mnt/data/em_image.png \
        --mask /mnt/data/em_image_seg.png \
        --outdir unet_em_run \
        --patch-size 256 \
        --stride 128 \
        --epochs 60 \
        --batch-size 8

This script:
1. loads the image and mask
2. extracts overlapping patches
3. splits patches into train/validation
4. trains a TensorFlow U-Net
5. predicts segmentation on the full image by tiled inference
6. saves the model, prediction, thresholded mask, and a preview figure
"""

import os
import math
import argparse
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------

def load_grayscale_image(path: str) -> np.ndarray:
    """Load image as float32 in [0,1], shape [H, W, 1]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., None]


def load_binary_mask(path: str, threshold: float = 0.5) -> np.ndarray:
    """Load mask as binary float32 {0,1}, shape [H, W, 1]."""
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.float32) / 255.0
    arr = (arr > threshold).astype(np.float32)
    return arr[..., None]


def save_uint8_image(path: str, arr: np.ndarray):
    """Save image, assuming arr is in [0,1] or already uint8."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (255.0 * arr).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ------------------------------------------------------------
# Patch extraction
# ------------------------------------------------------------

def compute_positions(length: int, patch_size: int, stride: int):
    """
    Return patch start positions covering the full axis.
    Ensures the last patch reaches the border.
    """
    if length <= patch_size:
        return [0]

    pos = list(range(0, length - patch_size + 1, stride))
    if pos[-1] != length - patch_size:
        pos.append(length - patch_size)
    return pos


def extract_patches(img: np.ndarray,
                    mask: np.ndarray,
                    patch_size: int = 256,
                    stride: int = 128,
                    min_fg_fraction: float = 0.0,
                    keep_empty_prob: float = 1.0):
    """
    Extract overlapping patches from image/mask.

    Parameters
    ----------
    min_fg_fraction:
        Always keep patches whose foreground fraction >= this value.
    keep_empty_prob:
        Probability to keep patches below min_fg_fraction, useful to reduce
        dominance of empty background patches.

    Returns
    -------
    X, Y: numpy arrays [N, patch_size, patch_size, 1]
    """
    h, w, _ = img.shape
    ys = compute_positions(h, patch_size, stride)
    xs = compute_positions(w, patch_size, stride)

    X = []
    Y = []

    for y in ys:
        for x in xs:
            ip = img[y:y + patch_size, x:x + patch_size, :]
            mp = mask[y:y + patch_size, x:x + patch_size, :]

            fg_fraction = float(mp.mean())

            if fg_fraction >= min_fg_fraction:
                keep = True
            else:
                keep = (np.random.rand() < keep_empty_prob)

            if keep:
                X.append(ip)
                Y.append(mp)

    if len(X) == 0:
        raise RuntimeError("No patches were extracted. Adjust patch size/stride/filtering.")

    X = np.stack(X, axis=0).astype(np.float32)
    Y = np.stack(Y, axis=0).astype(np.float32)
    return X, Y


# ------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------

def augment(image, mask):
    """TensorFlow augmentation. image/mask are [H,W,1]."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Mild intensity perturbation for image only
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def make_dataset(X, Y, batch_size=8, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if training:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

def conv_block(x, filters, dropout=0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def encoder_block(x, filters, dropout=0.0):
    c = conv_block(x, filters, dropout)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c)
    return c, p


def decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape=(256, 256, 1), base_filters=32):
    inputs = tf.keras.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters * 2)
    s3, p3 = encoder_block(p2, base_filters * 4)
    s4, p4 = encoder_block(p3, base_filters * 8, dropout=0.1)

    b = conv_block(p4, base_filters * 16, dropout=0.2)

    d1 = decoder_block(b, s4, base_filters * 8)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(d4)

    return tf.keras.Model(inputs, outputs, name="unet_binary_segmentation")


# ------------------------------------------------------------
# Losses and metrics
# ------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denom + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# ------------------------------------------------------------
# Train/validation split
# ------------------------------------------------------------

def split_train_val(X, Y, val_fraction=0.2, seed=42):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(val_fraction * n)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # fallback in pathological tiny cases
    if len(train_idx) == 0:
        train_idx = val_idx[:1]
        val_idx = val_idx[1:] if len(val_idx) > 1 else val_idx

    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


# ------------------------------------------------------------
# Tiled full-image inference
# ------------------------------------------------------------

def tiled_predict(model,
                  img: np.ndarray,
                  patch_size: int = 256,
                  stride: int = 128) -> np.ndarray:
    """
    Predict full-size probability map by overlapping tile inference
    with averaging in overlap regions.

    img: [H, W, 1], float32 in [0,1]
    returns: [H, W], float32 in [0,1]
    """
    h, w, _ = img.shape
    ys = compute_positions(h, patch_size, stride)
    xs = compute_positions(w, patch_size, stride)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_cnt = np.zeros((h, w), dtype=np.float32)

    for y in ys:
        for x in xs:
            patch = img[y:y + patch_size, x:x + patch_size, :]
            pred = model.predict(patch[None, ...], verbose=0)[0, :, :, 0]
            prob_sum[y:y + patch_size, x:x + patch_size] += pred
            prob_cnt[y:y + patch_size, x:x + patch_size] += 1.0

    prob = prob_sum / np.maximum(prob_cnt, 1e-6)
    return prob


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def save_preview_figure(path, image, mask, prob, pred_bin):
    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(image.squeeze(), cmap="gray")
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(mask.squeeze(), cmap="gray")
    ax2.set_title("Ground truth")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(prob, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Probability")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
    ax4.set_title("Thresholded")
    ax4.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a U-Net on grayscale image + binary mask.")

    p.add_argument("--image", type=str, required=True, help="Path to grayscale training image")
    p.add_argument("--mask", type=str, required=True, help="Path to binary training mask")
    p.add_argument("--outdir", type=str, default="unet_em_output", help="Output directory")

    p.add_argument("--patch-size", type=int, default=256, help="Patch size")
    p.add_argument("--stride", type=int, default=128, help="Patch extraction / inference stride")
    p.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for final binary mask")
    p.add_argument("--base-filters", type=int, default=32, help="Base number of U-Net filters")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # patch filtering to reduce too many empty background patches
    p.add_argument("--min-fg-fraction", type=float, default=0.001,
                   help="Always keep patches with at least this foreground fraction")
    p.add_argument("--keep-empty-prob", type=float, default=0.25,
                   help="Probability to keep patches with foreground below min-fg-fraction")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    print("Loading data...")
    image = load_grayscale_image(args.image)
    mask = load_binary_mask(args.mask)

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: {image.shape[:2]} vs {mask.shape[:2]}"
        )

    h, w, _ = image.shape
    print(f"Image shape: {image.shape}")
    print(f"Mask foreground fraction: {mask.mean():.6f}")

    if args.patch_size > h or args.patch_size > w:
        raise ValueError(
            f"Patch size {args.patch_size} is larger than image size {(h, w)}"
        )

    print("Extracting patches...")
    X, Y = extract_patches(
        image,
        mask,
        patch_size=args.patch_size,
        stride=args.stride,
        min_fg_fraction=args.min_fg_fraction,
        keep_empty_prob=args.keep_empty_prob,
    )

    print(f"Extracted patches: {len(X)}")
    print(f"Patch shape: {X.shape[1:]}")

    X_train, Y_train, X_val, Y_val = split_train_val(
        X, Y,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

    print(f"Training patches:   {len(X_train)}")
    print(f"Validation patches: {len(X_val)}")

    train_ds = make_dataset(X_train, Y_train, batch_size=args.batch_size, training=True)
    val_ds = make_dataset(X_val, Y_val, batch_size=args.batch_size, training=False)

    print("Building model...")
    model = build_unet(
        input_shape=(args.patch_size, args.patch_size, 1),
        base_filters=args.base_filters
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef, "binary_accuracy"]
    )

    model.summary()

    checkpoint_path = os.path.join(args.outdir, "best_unet.keras")
    history_path = os.path.join(args.outdir, "history.npz")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Saving final model...")
    model.save(os.path.join(args.outdir, "final_unet.keras"))

    print("Saving training history...")
    np.savez(history_path, **history.history)

    print("Running tiled full-image inference...")
    prob = tiled_predict(
        model,
        image,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    pred_bin = (prob > args.threshold).astype(np.float32)

    print("Saving outputs...")
    save_uint8_image(os.path.join(args.outdir, "input.png"), image.squeeze())
    save_uint8_image(os.path.join(args.outdir, "mask_gt.png"), mask.squeeze())
    save_uint8_image(os.path.join(args.outdir, "prediction_prob.png"), prob)
    save_uint8_image(os.path.join(args.outdir, "prediction_binary.png"), pred_bin)

    save_preview_figure(
        os.path.join(args.outdir, "preview.png"),
        image=image,
        mask=mask,
        prob=prob,
        pred_bin=pred_bin,
    )

    # quick full-image scores against available mask
    gt = mask.squeeze()
    pr = pred_bin

    intersection = np.sum((gt > 0.5) & (pr > 0.5))
    gt_sum = np.sum(gt > 0.5)
    pr_sum = np.sum(pr > 0.5)
    union = gt_sum + pr_sum - intersection

    dice = (2.0 * intersection + 1e-6) / (gt_sum + pr_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    acc = np.mean((gt > 0.5) == (pr > 0.5))

    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"Full-image Dice: {dice:.6f}\n")
        f.write(f"Full-image IoU:  {iou:.6f}\n")
        f.write(f"Full-image Acc:  {acc:.6f}\n")

    print(f"Full-image Dice: {dice:.6f}")
    print(f"Full-image IoU:  {iou:.6f}")
    print(f"Full-image Acc:  {acc:.6f}")
    print(f"Done. Results saved in: {args.outdir}")


if __name__ == "__main__":
    main()
