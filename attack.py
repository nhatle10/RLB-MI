import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from generator import Generator
from classify import *
from utils import *
from copy import deepcopy


def inversion(
    agent,
    G,
    T,
    E,
    alpha,
    z_dim=100,
    max_episodes=40000,
    max_step=1,
    label=0,
    model_name="VGG16",
):
    print("Target Label : " + str(label))
    best_score = 0

    # Add metrics for evaluation model E
    accuracy_E_list = []
    accuracy_top5_E_list = []

    for i_episode in range(1, max_episodes + 1):
        y = torch.tensor([label]).cuda()

        # Initialize the state at the beginning of each episode.
        z = torch.randn(1, z_dim).cuda()
        state = deepcopy(z.cpu().numpy())
        for t in range(max_step):

            # Update the state and create images from the updated state and action.
            action = agent.act(state)
            z = (
                alpha * z
                + (1 - alpha) * action.clone().detach().reshape((1, len(action))).cuda()
            )
            next_state = deepcopy(z.cpu().numpy())
            state_image = G(z).detach()
            action_image = G(
                action.clone().detach().reshape((1, len(action))).cuda()
            ).detach()

            # Calculate the reward.
            _, state_output = T(state_image)
            _, action_output = T(action_image)
            score1 = float(
                torch.mean(
                    torch.diag(
                        torch.index_select(
                            torch.log(F.softmax(state_output, dim=-1)).data, 1, y
                        )
                    )
                )
            )
            score2 = float(
                torch.mean(
                    torch.diag(
                        torch.index_select(
                            torch.log(F.softmax(action_output, dim=-1)).data, 1, y
                        )
                    )
                )
            )
            score3 = math.log(
                max(
                    1e-7,
                    float(
                        torch.index_select(F.softmax(state_output, dim=-1).data, 1, y)
                    )
                    - float(
                        torch.max(
                            torch.cat(
                                (
                                    F.softmax(state_output, dim=-1)[0, :y],
                                    F.softmax(state_output, dim=-1)[0, y + 1 :],
                                )
                            ),
                            dim=-1,
                        )[0]
                    ),
                )
            )
            reward = 2 * score1 + 2 * score2 + 8 * score3

            # Update policy.
            if t == max_step - 1:
                done = True
            else:
                done = False

            agent.step(state, action, reward, next_state, done, t)
            state = next_state

        # Save the image with the maximum confidence score.
        test_images = []
        test_scores = []
        for i in range(1):
            with torch.no_grad():
                z_test = torch.randn(1, z_dim).cuda()
                for t in range(max_step):
                    state_test = z_test.cpu().numpy()
                    action_test = agent.act(state_test)
                    z_test = (
                        alpha * z_test
                        + (1 - alpha)
                        * action_test.clone()
                        .detach()
                        .reshape((1, len(action_test)))
                        .cuda()
                    )
                test_image = G(z_test).detach()
                test_images.append(test_image.cpu())
                _, test_output = T(test_image)
                test_score = float(
                    torch.mean(
                        torch.diag(
                            torch.index_select(
                                F.softmax(test_output, dim=-1).data, 1, y
                            )
                        )
                    )
                )

            test_scores.append(test_score)
        mean_score = sum(test_scores) / len(test_scores)

        # Save metrics every 500 episodes
        if i_episode % 500 == 0:
            # Also evaluate using model E (FaceNet)
            with torch.no_grad():
                # Use low2high preprocessing for model E as in main.py
                test_image_E = low2high(torch.vstack(test_images))
                _, E_output = E(test_image_E)

                # Calculate top-1 and top-5 accuracy for evaluation model E
                E_softmax_output = F.softmax(E_output, dim=-1).data.cpu()
                _, E_top_indices = torch.topk(E_softmax_output, 5, dim=1)
                E_top1_correct = E_top_indices[:, 0] == y.cpu()
                E_top5_correct = (E_top_indices == y.cpu().unsqueeze(1)).any(dim=1)
                E_top1_accuracy = float(E_top1_correct.float().mean())
                E_top5_accuracy = float(E_top5_correct.float().mean())

            # Store metrics
            accuracy_E_list.append(E_top1_accuracy)
            accuracy_top5_E_list.append(E_top5_accuracy)

            # Save metrics to file
            metrics_dir = f"./result/metrics/{model_name}"
            os.makedirs(f"{metrics_dir}/{label}", exist_ok=True)
            metrics_file = f"{metrics_dir}/{label}/{alpha}.npz"
            np.savez(
                metrics_file,
                accuracy_E=np.array(accuracy_E_list),
                accuracy_top5_E=np.array(accuracy_top5_E_list),
            )
            print(f"Episode {i_episode}:")
            print(
                f"  Model E - Accuracy: {E_top1_accuracy:.4f}, Top-5 accuracy: {E_top5_accuracy:.4f}"
            )

        if mean_score >= best_score:
            best_score = mean_score
            best_images = torch.vstack(test_images)
            os.makedirs("./result/images/{}".format(model_name), exist_ok=True)
            os.makedirs("./result/models/{}".format(model_name), exist_ok=True)
            save_image(
                best_images,
                "./result/images/{}/{}_{}.png".format(model_name, label, alpha),
                nrow=10,
            )
            torch.save(
                agent.actor_local.state_dict(),
                "./result/models/{}/actor_{}_{}.pt".format(model_name, label, alpha),
            )

        if i_episode % 10000 == 0 or i_episode == max_episodes:
            print(
                "Episodes {}/{}, Confidence score for the target model : {:.4f}".format(
                    i_episode, max_episodes, best_score
                )
            )

    return best_images
