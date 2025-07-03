# DreamerV2 - PyTorch Implementation

This is a PyTorch implementation of the DreamerV2 algorithm. The project is currently configured to train an agent on the Atari games Breakout and Pong.

## Project Status

**This project is currently under development and is not yet complete.** There are known bugs and issues that need to be addressed before the model can be trained successfully.

## File Structure

Here is a breakdown of the important files and directories in this project:

```
├── configs/
│   ├── pong.yaml           # Configuration for the Pong environment
│   └── breakout.yaml       # Configuration for the Breakout environment
├── src/
│   ├── train.py            # Main script to start training
│   ├── envs/
│   │   └── atari.py        # Wrapper for Atari environments
│   ├── models/
│   │   ├── rssm.py         # Recurrent State-Space Model (RSSM)
│   │   ├── actor_critic.py # Actor and Critic networks
│   │   ├── vision.py       # Encoder and Decoder for vision
│   │   ├── heads.py        # Reward and discount prediction heads
│   │   └── behavior.py     # Agent's imagination and behavior learning
│   ├── replay/
│   │   └── replay_buffer.py # Replay buffer for storing experiences
│   └── utils/
│       ├── tools.py        # Helper functions (e.g., lambda_return)
│       ├── image_saver.py  # Utility for saving reconstructed images
│       └── perceptual_loss.py # VGG perceptual loss implementation (an idea but is commented)
└── README.md               # This file
```

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy pyyaml wandb gymnasium tqdm ale-py opencv-python
    ```

2.  **Start training:**
    ```bash
    python -m src.train --config configs/breakout.yaml
    ```

## Known Issues

*   Recon loss converges too fast
*   Hyperparameters may not be fully optimized.
