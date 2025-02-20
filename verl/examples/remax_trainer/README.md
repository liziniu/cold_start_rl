
# ReMax Training 


This repo contains the code for training through ReMax.

We make the following changes to the original ReMax code in the Verl repo:

1. We improve the ReMax's training effiency by a better implementation of the greedy decoding.
2. We use a cosine learning rate scheduler rather than a constant learning rate.
3. We do not use the kl-regularization loss in the online RL training.
4. We remove the weight decay in Adam's optimizer and set betas to (0.9, 0.95).
5. We remove the entropy regularization in RL loss.
6. We add a new `gsm8k_boxed` that follows the reward function with the `math` dataset. 

Note that hyper-parameters are not carefully tuned.








