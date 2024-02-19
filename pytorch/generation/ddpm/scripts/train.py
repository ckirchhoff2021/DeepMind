import argparse
import datetime
import torch
import wandb
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ddpm.script_utils as script_utils
import cv2
from torchvision.utils import save_image
import torch.nn.functional as F


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=args.img_channels,

        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else 10,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (32, 32), args.img_channels, 10,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion


def main():
    args = create_argparser().parse_args()
    device = args.device
    print("===== args: ", args.img_channels)

    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

    if args.model_checkpoint is not None:
        diffusion.load_state_dict(torch.load(args.model_checkpoint))
    if args.optim_checkpoint is not None:
        optimizer.load_state_dict(torch.load(args.optim_checkpoint))

    batch_size = args.batch_size

    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='/home/cx/datas',
        train=True,
        download=False,
        transform=image_transform,
    )

    test_dataset = datasets.MNIST(
        root='/home/cx/datas',
        train=False,
        download=False,
        transform=image_transform,
    )

    train_loader = script_utils.cycle(DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    ))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4)

    acc_train_loss = 0

    for iteration in range(1, args.iterations + 1):
        diffusion.train()

        x, y = next(train_loader)
        x = x.to(device)
        y = y.to(device)

        if args.use_labels:
            loss = diffusion(x, y)
        else:
            loss = diffusion(x)

        acc_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        diffusion.update_ema()

        if iteration % args.log_rate == 0:
            test_loss = 0
            with torch.no_grad():
                diffusion.eval()
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    if args.use_labels:
                        loss = diffusion(x, y)
                    else:
                        loss = diffusion(x)

                    test_loss += loss.item()

            if args.use_labels:
                samples = diffusion.sample(100, device, y=torch.arange(10, device=device))
            else:
                samples = diffusion.sample(100, device)

            # samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
            samples = ((samples + 1) / 2).clip(0, 1)

            test_loss /= len(test_loader)
            acc_train_loss /= args.log_rate

            print(
                "==> iteration: [%d], test_loss = %.3f, acc_train_loss = %.3f" % (iteration, test_loss, acc_train_loss))
            print("== samples: ", samples.shape)
            save_image(samples, os.path.join(args.log_dir, f"samples/{iteration}.png"), nrow=10)

            acc_train_loss = 0

        if iteration % args.checkpoint_rate == 0:
            model_filename = f"{args.log_dir}/models/iteration-{iteration}-model.pth"
            optim_filename = f"{args.log_dir}/models/iteration-{iteration}-optim.pth"

            torch.save(diffusion.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), optim_filename)


def create_argparser():
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=256,
        iterations=800000,

        log_rate=100,
        checkpoint_rate=1000,
        log_dir="mnist",

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())
    defaults.update(dict(img_channels=1))
    # print(defaults)

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
