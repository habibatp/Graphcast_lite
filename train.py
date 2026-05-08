# graphcast_lite/train.py
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import config

from dataset import create_dataloaders
from graphcast_lite import GraphCastLite, ModelConfig, TaskConfigLite


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def train_one_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        x, f, y = batch

        x = x.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.USE_AMP):
            loss_per_batch, diagnostics = model.loss(
                inputs=x,
                targets=y,
                forcings=f,
            )

            loss = loss_per_batch.mean()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(val_loader, desc="Validation")

    for batch in pbar:
        x, f, y = batch

        x = x.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss_per_batch, diagnostics = model.loss(
            inputs=x,
            targets=y,
            forcings=f,
        )

        loss = loss_per_batch.mean()
        total_loss += loss.item()

        pbar.set_postfix({"val_loss": loss.item()})

    return total_loss / len(val_loader)


def main():
    device = torch.device(config.DEVICE)

    print("Device:", device)

    train_loader, val_loader = create_dataloaders(
        data_dir=config.DATA_DIR,
        stats_dir=config.STATS_DIR,
        input_steps=config.INPUT_STEPS,
        target_lead_times=config.TARGET_LEAD_TIMES,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        include_forcings=True,
    )

    model_config = ModelConfig(
        resolution=1.0,
        mesh_size=config.MESH_SIZE,
        latent_size=config.LATENT_SIZE,
        gnn_msg_steps=config.GNN_MSG_STEPS,
        hidden_layers=config.HIDDEN_LAYERS,
        radius_query_fraction_edge_length=config.RADIUS_QUERY_FRACTION_EDGE_LENGTH,
    )

    task_config = TaskConfigLite(
        input_channels=config.INPUT_CHANNELS,
        forcing_channels=config.FORCING_CHANNELS,
        output_channels=config.OUTPUT_CHANNELS,
        input_steps=config.INPUT_STEPS,
    )

    model = GraphCastLite(
        model_config=model_config,
        task_config=task_config,
        lat=config.LAT,
        lon=config.LON,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scaler = GradScaler(enabled=config.USE_AMP)

    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{config.EPOCHS} =====")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        print(f"Train loss: {train_loss:.6f}")
        print(f"Val loss  : {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = os.path.join(
                config.CHECKPOINT_DIR,
                "best_graphcast_lite.pt",
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                path=best_path,
            )

            print("✅ Best checkpoint saved:", best_path)

        if epoch % config.SAVE_EVERY == 0:
            epoch_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"graphcast_lite_epoch_{epoch}.pt",
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                path=epoch_path,
            )

            print("✅ Epoch checkpoint saved:", epoch_path)


if __name__ == "__main__":
    main()