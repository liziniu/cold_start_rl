# RL with Better Cold-Start Strategies

This repository implements training methods using GEM and ReMax to enhance the reasoning capabilities of Large Language Models (LLMs).

For an in-depth analysis of our methodology and findings, please refer to our blog post: "Can Better Cold-Start Strategies Improve RL Training for LLMs?" at https://tangible-polo-203.notion.site/Can-Better-Cold-Start-Strategies-Improve-RL-Training-for-LLMs-17aa0742a51680828616c867ed53bc6b

```
TL;DR

Reinforcement Learning (RL) plays a crucial role in enhancing Large Language Models (LLMs) for complex tasks like mathematical reasoning. However, when cold-starting LLMs for RL training, the challenge lies in maintaining output diversity. Traditional supervised fine-tuning (SFT) for cold-start can lead to overfitting, reducing output diversity and restrict the performance improvement in later RL stages. This blog discusses a new approach using GEM, a diversity-preserving fine-tuning algorithm, which helps optimize the RL training process.
```


Project artifacts and training logs are available at: https://wandb.ai/liziniu1997/verl_remax_openr1_training

## Requirements

Core dependencies:
```
vllm==0.6.3
torch==2.4.0
transformers==4.47.1
```
Additional dependencies can be found in the `verl` package.

## Project Structure

- **GEM Training**: Implementation and scripts are located in the `GEM` directory. See `GEM/scripts` for training examples and documentation.

- **ReMax Training**: Implementation and scripts are available in the `verl/examples/remax_trainer` directory.


## Citation

If you find this work useful, please cite the following paper:

```
@inproceedings{li2025preserving,
  title={Preserving Diversity in Supervised Fine-Tuning of Large Language Models},
  author={Li, Ziniu and Chen, Congliang and Xu, Tian and Qin, Zeyu and Xiao, Jiancong and Sun, Ruoyu and Luo, Zhi-Quan},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

```
@inproceedings{li2024remax,
  title={Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models},
  author={Li, Ziniu and Xu, Tian and Zhang, Yushun and Lin, Zhihang and Yu, Yang and Sun, Ruoyu and Luo, Zhi-Quan},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```







