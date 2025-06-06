import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from generator import Generator
from classify import *
from utils import *
from SAC import Agent
from attack import inversion
import pickle

parser = argparse.ArgumentParser(description="RLB-MI")
parser.add_argument("-model_name", default="VGG16")
parser.add_argument("-max_episodes", type=int, default=40000)
parser.add_argument("-max_step", type=int, default=1)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-alpha", type=float, default=0)
parser.add_argument("-n_classes", type=int, default=1000)
parser.add_argument("-z_dim", type=int, default=100)
parser.add_argument("-n_target", type=int, default=100)
parser.add_argument(
    "-start", type=int, default=0, help="Start index for target classes"
)
parser.add_argument("-end", type=int, default=100, help="End index for target classes")
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model_name
    max_episodes = args.max_episodes
    max_step = args.max_step
    seed = args.seed
    alpha = args.alpha
    n_classes = args.n_classes
    z_dim = args.z_dim
    n_target = args.n_target

    print("Target Model : " + model_name)
    G = Generator(z_dim)
    G = nn.DataParallel(G).cuda()
    G = G.cuda()
    ckp_G = torch.load("/kaggle/input/weightcs106/weights/CelebA.tar")["state_dict"]
    load_my_state_dict(G, ckp_G)
    G.eval()

    if model_name == "VGG16":
        T = VGG16(n_classes)
        path_T = "/kaggle/input/weightcs106/weights/VGG16.tar"
    elif model_name == "ResNet-152":
        T = IR152(n_classes)
        path_T = "/kaggle/input/weightcs106/weights/ResNet-152.tar"
    elif model_name == "Face.evoLVe":
        T = FaceNet64(n_classes)
        path_T = "/kaggle/input/weightcs106/weights/Face.evoLVe.tar"

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T["state_dict"], strict=False)
    T.eval()

    E = FaceNet(n_classes)
    path_E = "/kaggle/input/weightcs106/weights/Eval.tar"
    E = torch.nn.DataParallel(E).cuda()
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E["state_dict"], strict=False)
    E.eval()

    def seed_everything(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

    seed_everything(seed)

    # Initialize arrays to store accuracies for each checkpoint
    num_checkpoints = (max_episodes // 500) + 1
    cnt = [0] * num_checkpoints  # Use proper initialization
    cnt5 = [0] * num_checkpoints

    identities = range(n_classes)
    targets = list(identities)[:n_target]
    targets = targets[args.start : args.end]

    for i in targets:
        agent = Agent(
            state_size=z_dim,
            action_size=z_dim,
            random_seed=seed,
            hidden_size=256,
            action_prior="uniform",
        )
        # Get both image and accuracy lists
        recon_image, accuracy_list, accuracy_top5_list = inversion(
            agent,
            G,
            T,
            E,
            alpha,
            z_dim=z_dim,
            max_episodes=max_episodes,
            max_step=max_step,
            label=i,
            model_name=model_name,
        )

        # Add the accuracies from each checkpoint to the running totals
        for idx in range(len(accuracy_list)):
            if accuracy_list[idx] > 0:  # If this example was correctly classified
                cnt[idx] += 1
            if accuracy_top5_list[idx] > 0:  # If this example was in top-5
                cnt5[idx] += 1

        print(
            "Classes {}, Number of correct target Top-1 : {}, Number of correct target Top-5 : {}".format(
                i, cnt[-1], cnt5[-1]
            )
        )

    # At the end, print or save the progression of accuracy across checkpoints
    for idx, episode in enumerate(range(0, max_episodes + 1, 500)):
        print(
            f"After episode {episode}: Top-1 accuracy: {cnt[idx]}/{len(targets)}, "
            f"Top-5 accuracy: {cnt5[idx]}/{len(targets)}"
        )

    with open("top1_acc.pkl", "wb") as f:
        pickle.dump(cnt, f)

    with open("top5_acc.pkl", "wb") as f:
        pickle.dump(cnt5, f)
