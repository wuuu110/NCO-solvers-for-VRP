# A Repository of the Studies Addressing Vehicle Routing Problems Using Neural Combinatorial Optimization Solvers 
This repository provides an up-to-date list of studies addressing Vehicle Routing Problems (VRPs) using Neural Combinatorial Optimization (NCO) solvers. It follows the taxonomy provided in our manuscript "[Neural Combinatorial Optimization Algorithms for Solving Vehicle Routing Problems: A Comprehensive Survey with Perspectives](https://arxiv.org/abs/2406.00415)". We are dedicated to updating this repository on a monthly basis. Your participation is appreciated; please consider starring ⭐️ this repository to stay informed about the latest updates and cite our paper if you find this repository beneficial 🚀🚀🚀.

```
@misc{wu_2024_neural,
      title={Neural Combinatorial Optimization Algorithms for Solving Vehicle Routing Problems: A Comprehensive Survey with Perspectives}, 
      author = {Wu, Xuan and Wang, Di and Wen, Lijie and Xiao, Yubin and Wu, Chunguo and Wu, Yuesong and Yu, Chaoyu and Maskell, Douglas L. and Zhou, You},
      year={2024},
      eprint={2406.00415},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
# News 📰
- **2025.05**: Our paper "Efficient Heuristics Generation for Solving Combinatorial Optimization Problems Using Large Language Models" is accepted by **SIGKDD 2025** [[paper](https://arxiv.org/abs/2505.12627)] [[code](https://github.com/wuuu110/Hercules)]. 
- **2025.05**: Our paper "DGL: Dynamic Global-Local Information Aggregation for Scalable VRP Generalization with Self-Improvement Learning" is accepted by **IJCAI 2025**. 
- **2025.03**: Our paper "Improving Generalization of Neural Vehicle Routing Problem Solvers Through the Lens of Model Architecture" is accepted by **Neural Networks** [[paper](https://arxiv.org/pdf/2406.06652)] [[code](https://github.com/xybFight/VRP-Generalization)] 
- **2024.11**: Our paper "An Efficient Diffusion-based Non-Autoregressive Solver for Traveling Salesman Problem" is accepted by **SIGKDD 2025** [[paper](https://arxiv.org/abs/2501.13767)] [[code](https://github.com/DEITSP/DEITSP)]
- **2024.10**: Our paper "Reinforcement Learning-based Non-Autoregressive
Solver for Traveling Salesman Problems" is accepted by **IEEE TNNLS**  [[paper](https://arxiv.org/pdf/2308.00560)] [[code](https://github.com/xybFight/NAR4TSP)]
- **2024.06**: We are excited to release this survey! 🚀
- **2023.12**: Our paper "Distilling Autoregressive Models to Obtain High-Performance Non-autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed" is accepted by **AAAI 2024** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/30008)] [[code](https://github.com/xybFight/GNARKD)]



# How To Request the Addition of a Paper 🤖
If you know any paper addressing VRPs with NCO solvers that is absent from this repository, please feel free to request its inclusion by either submitting a pull request or contacting us via email at  ([wuuu22@mails.jlu.edu.cn](wuuu22@mails.jlu.edu.cn) or [n2308769c@e.ntu.edu.sg]()).

# Timeline 🌟

The development of NCO solvers for VRPs. 
![Image text](https://github.com/wuuu110/NCO-solvers-for-VRP/blob/main/img-folder/timeline.jpg)

These papers are gathered from Google Scholar and Web of Science with the keywords "Neural Combinatorial Optimization" *OR* "NCO" *OR* "Reinforcement Learning" *OR* "Deep Learning" *OR* "Neural Network" *AND* "Vehicle Routing Problem" *OR* "VRP" *OR* "Traveling Salesman Problem" *OR* "TSP" by the end of 2023. Following the initial data collection, a meticulous examination of each paper is conducted to precisely define its scope within the realm of NCO.

In addition, these papers are meticulously organized in ascending order based on their respective publication years. Furthermore, within the same publication year, papers are sorted in ascending order, discerned by the initial letter of each respective title.

# Table of Contents (Adhering to the Structure Presented in the Paper) 📚
[Learning to Construct (L2C) solvers](#Learning-to-Construct-L2C-solvers)

[Learning to Improve (L2I) solvers](#Learning-to-Improve-L2I-solvers)

[Learning to Predict-Once (L2P-O) solvers](#Learning-to-Predict-Once-L2P-O-solvers)

[Learning to Predict-Multiplicity (L2C-M) solvers](#Learning-to-Predict-Multiplicity-L2C-M-solvers)

[NCO Solvers for Solving Out-of-distribution](#NCO-solvers-for-Solving-Out-of-distribution)

[NCO Solvers for Solving Large-scale VRPs](#NCO-solvers-for-Solving-Large-scale-VRPs)

[NCO Solvers for Solving Multi-constrained VRP Variants](##NCO-solvers-for-Solving-Multi-constrained-VRP-Variants)

## Learning to Construct (L2C) Solvers
&bull; Pointer Networks, ***NeurIPS***, 2015, [[paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)]

&bull; Neural Combinatorial Optimization with Reinforcement Learning, ***ICLR(workshop)***, 2017, [[paper](https://openreview.net/pdf?id=Bk9mxlSFx)]

&bull; Learning Combinatorial Optimization Algorithms over Graphs, ***NeurIPS***, 2017, [[paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf)]

&bull; Reinforcement Learning for Solving the Vehicle Routing Problem, ***NeurIPS***, 2018, [[paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/9fb4651c05b2ed70fba5afe0b039a550-Paper.pdf)]

&bull; Attention, Learn to Solve Routing Problems!, ***ICLR***, 2019, [[paper](https://openreview.net/pdf?id=ByxBFsRqYm)]

&bull; POMO: Policy Optimization with Multiple Optima for Reinforcement Learning, ***NeurIPS***, 2020, [[paper](https://proceedings.neurips.cc/paper/2020/file/f231f2107df69eab0a3862d50018a9b2-Paper.pdf)]

&bull; A Hybrid of Deep Reinforcement Learning and Local Search for the Vehicle Routing Problems, ***TITS***, 2021, [[paper](https://ieeexplore.ieee.org/document/9141401)]

&bull; Learning Collaborative Policies to Solve NP-hard Routing Problems, *NeurIPS*, 2021, [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/564127c03caab942e503ee6f810f54fd-Paper.pdf)]

&bull; Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems, ***AAAI***, 2021, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17430)]

&bull; Matrix Encoding Networks for Neural Combinatorial Optimization, ***NeurIPS***, 2021, [[paper](https://openreview.net/pdf?id=C__ChZs8WjU)]

&bull; Step-Wise Deep Learning Models for Solving Routing Problems, ***TII***, 2021, [[paper](https://ieeexplore.ieee.org/document/9226142)]

&bull; Deep Reinforcement Learning for Combinatorial Optimization: Covering Salesman Problems, ***TCYB***, 2022, [[paper](https://ieeexplore.ieee.org/document/9523517)]

&bull; Efficient Active Search for Combinatorial Optimization Problems, ***ICLR***, 2022, [[paper](https://openreview.net/pdf?id=nO5caZwFwYu)]

&bull; Heterogeneous attentions for solving pickup and delivery problem via deep reinforcement learning, ***TNNLS***, 2022, [[paper](https://ieeexplore.ieee.org/abstract/document/9352489)]

&bull; Reinforcement Learning With Multiple Relational Attention for Solving Vehicle Routing Problems, ***TCYB***, 2022, [[paper](https://ieeexplore.ieee.org/document/9478307)]

&bull; Scale-conditioned Adaptation for Large Scale Combinatorial Optimization, ***NeurIPS(workshop)***, 2022, [[paper](https://openreview.net/pdf?id=oy8hDBI8Qx)]

&bull; Simulation-guided Beam Search for Neural Combinatorial Optimization, ***NeurIPS***, 2022, [[paper](https://openreview.net/pdf?id=tYAS1Rpys5)]

&bull; Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization, ***NeurIPS***, 2022, [[paper](https://openreview.net/forum?id=kHrE2vi5Rvs)]

&bull; A Lightweight CNN-Transformer Model for Learning Traveling Salesman Problems, ***arXiv***, 2023, [[paper](https://arxiv.org/pdf/2305.01883.pdf)]

&bull; Combinatorial Optimization with Policy Adaptation using Latent Space Search, ***NeurIPS***, 2023, [[paper](https://openreview.net/pdf?id=vpMBqdt9Hl)]

&bull; Data-efficient Supervised Learning is Powerful for Neural Combinatorial Optimization, ***arXiv***, 2023, [[paper](https://openreview.net/pdf?id=a_yFkJ4-uEK)]

&bull; Learning Feature Embedding Refiner for Solving Vehicle Routing Problems, ***TNNLS***, 2023, [[paper](https://ieeexplore.ieee.org/document/10160045)]

&bull; Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization, ***NeurIPS***, 2023, [[paper](https://openreview.net/pdf?id=RBI4oAbdpm)]

&bull; RL-CSL: A Combinatorial Optimization Method Using Reinforcement Learning and Contrastive Self-Supervised Learning, ***TETCI***, 2023, [[paper](https://ieeexplore.ieee.org/document/9690950)]

&bull; SplitNet: A Reinforcement Learning Based Sequence Splitting Method for the MinMax Multiple Travelling Salesman Problem, ***AAAI***, 2023, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26049)]

&bull; Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems, ***arXiv***, 2023, [[paper](https://arxiv.org/abs/2308.00560)]

&bull; 2D-Ptr: 2D Array Pointer Network for Solving the Heterogeneous Capacitated Vehicle Routing Problem, ***AAMAS***, 2024, [[paper](https://dl.acm.org/doi/pdf/10.5555/3635637.3662981)]

&bull; Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed, ***AAAI***, 2024, [[paper](https://arxiv.org/abs/2312.12469)]

&bull; DPN: Decoupling Partition and Navigation for Neural Solvers of Min-max Vehicle Routing Problems, ***ICML***, 2024, [[paper](https://openreview.net/forum?id=ar174skI9u)]

&bull; Equity-Transformer: Solving NP-hard Min-Max Routing Problems as Sequential Generation with Equity Context, ***AAAI***, 2024, [[paper](https://arxiv.org/abs/2306.02689)]

&bull; Hierarchical Neural Constructive Solver for Real-world TSP Scenarios, ***SIGKDD***, 2024, [[paper](https://www.arxiv.org/pdf/2408.03585)]

&bull; Learning Encodings for Constructive Neural Combinatorial Optimization Needs to Regret, ***AAAI***, 2024, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/30069)]

&bull; Leader Reward for POMO-Based Neural Combinatorial Optimization, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2405.13947)]

&bull; Learning to Handle Complex Constraints for Vehicle Routing Problems,***NeurIPS***, 2024, [[paper](https://nips.cc/virtual/2024/poster/95638)]

&bull; Neural Combinatorial Optimization for Robust Routing Problem with Uncertain Travel Times, ***NeurIPS***, 2024, [[paper](https://nips.cc/virtual/2024/poster/96075)]

&bull; PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2402.14048.pdf)]

&bull; Towards Generalizable Neural Solvers for Vehicle Routing Problems via Ensemble with Transferrable Local Policy, ***IJCAI***, 2024, [[paper](https://arxiv.org/pdf/2308.14104.pdf)]

## Learning to Improve (L2I) Solvers

&bull; Learning to Perform Local Rewriting for Combinatorial Optimization, ***NeurIPS***, 2019, [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf)]

&bull; A Learning-based Iterative Method for Solving Vehicle Routing Problems, ***ICLR***, 2020, [[paper](https://openreview.net/pdf?id=BJe1334YDH)]

&bull; Learning 2-opt Heuristics for the Traveling Salesman Problem via Deep Reinforcement Learning, ***ACML***, 2020, [[paper](https://proceedings.mlr.press/v129/costa20a/costa20a.pdf)]

&bull; Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem, ***ECAI***, 2020, [[paper](https://ecai2020.eu/papers/786_paper.pdf)]

&bull; Learning 3-opt Heuristics for Traveling Salesman Problem via Deep Reinforcement Learning, ***ACML***, 2021, [[paper](https://proceedings.mlr.press/v157/sui21a/sui21a.pdf)]

&bull; Learning Improvement Heuristics for Solving Routing Problems, ***TNNLS***, 2021, [[paper](https://ieeexplore.ieee.org/document/9393606)]

&bull; Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer, ***NeurIPS***, 2021, [[paper](https://proceedings.neurips.cc/paper/2021/file/5c53292c032b6cb8510041c54274e65f-Paper.pdf)]

&bull; Effcient Neural Neighborhood Search for Pickup and Delivery Problems, ***IJCAI***, 2022, [[paper](https://www.ijcai.org/proceedings/2022/0662.pdf)]

&bull; Learning to CROSS Exchange to Solve Min-max Vehicle Routing Problems, ***ICLR***, 2023, [[paper](https://openreview.net/pdf?id=ZcnzsHC10Y)]

&bull; Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-opt, ***NeurIPS***, 2023, [[paper](https://openreview.net/pdf?id=q1JukwH2yP)]

&bull; Select and Optimize: Learning to Solve Large-scale TSP Instances, ***AISTATS***, 2023, [[paper](https://proceedings.mlr.press/v206/cheng23a/cheng23a.pdf)]

&bull; Destroy and Repair Using Hyper-Graphs for Routing, ***AAAI***, 2025, [[paper](https://www.arxiv.org/pdf/2502.16170)]


## Learning to Predict-Once (L2P-O) Solvers

&bull; An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem, ***arXiv***, 2019, [[paper](https://arxiv.org/pdf/1906.01227.pdf)]

&bull; Learning to Solve NP-Complete Problems: A Graph Neural Network for Decision TSP, ***AAAI***, 2019, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4399)]

&bull; Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances, ***AAAI***, 2021, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16916)]

&bull; Generalization of Machine Learning for Problem Reduction: A Case Study on Travelling Salesman Problems, ***OR Spectrum***, 2021, [[paper](https://link.springer.com/article/10.1007/s00291-020-00604-x)]

&bull; NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem, ***NeurIPS***, 2021, [[paper](https://proceedings.neurips.cc/paper/2021/file/3d863b367aa379f71c7afc0c9cdca41d-Paper.pdf)]

&bull; Combining Reinforcement Learning and Optimal Transport for the Traveling Salesman Problem, ***arXiv***, 2022, [[paper](https://arxiv.org/pdf/2203.00903.pdf)]

&bull; DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems, ***NeurIPS***, 2022, [[paper](https://openreview.net/pdf?id=9u05zr0nhx)]

&bull; Deep Policy Dynamic Programming for Vehicle Routing Problems, ***CPAIOR***, 2022, [[paper](https://link.springer.com/chapter/10.1007/978-3-031-08011-1_14)]

&bull; Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time, ***ICLR***, 2023, [[paper](https://openreview.net/pdf?id=6ZajpxqTlQ)]

&bull; Graph Neural Network Guided Local Search for the Traveling Salesperson Problem, ***ICLR***, 2022, [[paper](https://openreview.net/pdf?id=ar92oEosBIg)]

&bull; DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization, ***NeurIPS***, 2023, [[paper](https://arxiv.org/pdf/2302.08224.pdf)]

&bull; DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization, ***NeurIPS***, 2023, [[paper](https://arxiv.org/pdf/2309.14032.pdf)]

&bull; From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization, ***NeurIPS***, 2023, [[paper](https://openreview.net/pdf?id=JtF0ugNMv2)]

&bull; Unsupervised Learning for Solving the Travelling Salesman Problem, ***NeurIPS***, 2023, [[paper](https://openreview.net/pdf?id=lAEc7aIW20)]

&bull; An Edge-Aware Graph Autoencoder Trained on Scale-Imbalanced Data for Traveling Salesman Problems, ***KBS***, 2024, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124001941)]

&bull; Ant Colony Sampling with GFlowNets for Combinatorial Optimization, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2403.07041.pdf)]

&bull; Fast T2T: Optimization Consistency Speeds Up Diffusion-Based Training-to-Testing Solving for Combinatorial Optimization, ***NeurIPS***, 2024, [[paper](https://openreview.net/forum?id=xDrKZOZEOc&noteId=7cfZRv82pZ)]


&bull; DISCO: Efficient Diffusion Solver for Large-Scale Combinatorial Optimization Problems, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2406.19705)]

&bull; GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time, ***AAAI***, 2024, [[paper](https://arxiv.org/pdf/2312.08224.pdf)]

&bull; On Size and Hardness Generalization in Unsupervised Learning for the Travelling Salesman Problem, ***arXiv***, 2024, [[paper](https://arxiv.org/abs/2403.20212)]

&bull; OptCM: The Optimization Consistency Models for Solving Combinatorial Problems in Few Shots, ***NeurIPS***, 2024, [[paper](https://neurips.cc/virtual/2024/poster/93096)]

&bull; Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling Salesman Problems, ***ICML***, 2024, [[paper](https://arxiv.org/pdf/2406.03503)]

&bull; An Efficient Diffusion-based Non-Autoregressive Solver for Traveling Salesman Problem, ***SIGKDD***, 2025, [[paper](https://github.com/DEITSP/DEITSP)]


## Learning to Predict-Multiplicity (L2C-M) Solvers

&bull; Boosting Dynamic Programming with Neural Networks for Solving NP-hard Problems, ***ACML***, 2018, [[paper](https://proceedings.mlr.press/v95/yang18a/yang18a.pdf)]

&bull; Approximate Dynamic Programming with Neural Networks in Linear Discrete Action Spaces, ***arXiv***, 2019, [[paper](https://arxiv.org/pdf/1902.09855.pdf)]

&bull; Deep Neural Network Approximated Dynamic Programming for Combinatorial Optimization, ***AAAI***, 2020, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5531)]

&bull; Combining Reinforcement Learning with Lin-Kernighan-Helsgaun Algorithm for the Traveling Salesman Problem, ***AAAI***, 2021, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17476)]

&bull; Learning a Latent Search Space for Routing Problems using Variational Autoencoders, ***ICLR***, 2021, [[paper](https://openreview.net/pdf?id=90JprVrJBO)]

&bull; Learning to Delegate for Large-scale Vehicle Routing ***NeurIPS***, 2021, [[paper](https://openreview.net/pdf?id=rm0I5y2zkG8)]

&bull; RBG: Hierarchically Solving Large-Scale Routing Problems in Logistic Systems via Reinforcement Learning, ***SIGKDD***, 2022, [[paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539037)]

&bull; Algorithm Evolution Using Large Language Model, ***arXiv***, 2023, [[paper](https://arxiv.org/pdf/2311.15249.pdf)]

&bull; Evolution of heuristics: Towards efficient automatic algorithm design using large language model, ***ICML***, 2024, [[paper](https://arxiv.org/pdf/2401.02051.pdf)]

&bull; Large Language Models as Hyper-Heuristics for Combinatorial Optimization, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2402.01145.pdf)]

&bull; MOCO: A Learnable Meta Optimizer for Combinatorial Optimization, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2402.04915.pdf)]

## NCO Solvers for Solving Out-of-distribution

&bull; Generalization of Neural Combinatorial Solvers Through the Lens of Adversarial Robustness, ***ICLR***, 2022, [[paper](https://openreview.net/pdf?id=vJZ7dPIjip3)]

&bull; Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation, ***NeurIPS***, 2022, [[paper](https://openreview.net/pdf?id=sOVNpUEgKMp)]

&bull; Learning to Solve Routing Problems via Distributionally Robust Optimization, ***AAAI***, 2022, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21214)]

&bull; Learning to Solve Travelling Salesman Problem with Hardness-adaptive Curriculum, ***AAAI***, 2022, [[paper](https://arxiv.org/abs/2204.03236)]

&bull; On the Generalization of Neural Combinatorial Optimization Heuristics, ***ECMLPKDD***, 2022, [[paper]([https://arxiv.org/abs/2204.03236](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_26))]

&bull; Multi-view graph contrastive learning for solving vehicle routing problems, ***UAI***, 2023, [[paper](https://proceedings.mlr.press/v216/jiang23a/jiang23a.pdf)]

&bull; Towards Omni-generalizable Neural Methods for Vehicle Routing Problems, ***ICML***, 2023, [[paper](https://proceedings.mlr.press/v202/zhou23o/zhou23o.pdf)]

&bull; INViT: A Generalizable Routing Problem Solver with Invariant Nested View Transformer, ***ICML***, 2024, [[paper](https://arxiv.org/pdf/2402.02317)]

&bull; Improving Generalization of Neural Vehicle Routing Problem Solvers Through the Lens of Model Architecture, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2406.06652)]

&bull; Towards Generalizable Neural Solvers for Vehicle Routing Problems via Ensemble with Transferrable Local Policy, ***IJCAI***, 2024, [[paper](https://arxiv.org/pdf/2308.14104.pdf)]

## NCO Solvers for Solving Large-scale VRPs
&bull; Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances, ***AAAI***, 2021, [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16916)]

&bull; Learning to Delegate for Large-scale Vehicle Routing ***NeurIPS***, 2021, [[paper](https://openreview.net/pdf?id=rm0I5y2zkG8)]

&bull; DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems, ***NeurIPS***, 2022, [[paper](https://openreview.net/pdf?id=9u05zr0nhx)]

&bull; RBG: Hierarchically Solving Large-Scale Routing Problems in Logistic Systems via Reinforcement Learning, ***SIGKDD***, 2022, [[paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539037)]

&bull; DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization, ***NeurIPS***, 2023, [[paper](https://arxiv.org/pdf/2302.08224.pdf)]

&bull; Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time, ***ICLR***, 2023, [[paper](https://openreview.net/pdf?id=6ZajpxqTlQ)]

&bull; H-TSP: Hierarchically Solving the Large-Scale Travelling Salesman Problem, ***AAAI***, 2023, [[paper](https://arxiv.org/pdf/2304.09395.pdf)]

&bull; Memory-efficient Transformer-based network model for Traveling Salesman Problem, ***Neural Networks***, 2023, [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023000771)]

&bull; Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization, ***NeurIPS***, 2023, [[paper](https://openreview.net/pdf?id=RBI4oAbdpm)]

&bull; Pointerformer: Deep Reinforced Multi-Pointer Transformer for the Traveling Salesman Problem, ***AAAI***, 2023, [[paper](https://arxiv.org/pdf/2304.09407.pdf)]

&bull; CADO: Cost-Aware Diffusion Solvers for Combinatorial Optimization through RL fine-tuning, ***ICML workshop***, 2024, [[paper](https://openreview.net/forum?id=RRbwBbYcvK&noteId=YTtrLSf0uU)]

&bull; DISCO: Efficient Diffusion Solver for Large-Scale Combinatorial Optimization Problems, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2406.19705)]

&bull; GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time, ***AAAI***, 2024, [[paper](https://arxiv.org/pdf/2312.08224.pdf)]

&bull; Instance-Conditioned Adaptation for Large-scale Generalization of Neural Combinatorial Optimization, ***arXiv***, 2024, [[paper](https://arxiv.org/pdf/2405.01906)]

&bull; Self-Improvement for Neural Combinatorial Optimization: Sample Without Replacement, but Improvement, ***TMLR***, 2024, [[paper](https://openreview.net/pdf?id=agT8ojoH0X)]

&bull; Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization, ***ECAI***, 2024, [[paper](https://arxiv.org/abs/2407.17206)]

&bull; UDC: A Unified Neural Divide-and-Conquer Framework for Large-Scale Combinatorial Optimization Problems, ***NeurIPS***, 2024, [[paper](https://arxiv.org/pdf/2407.00312)]

&bull; Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems, ***ICLR***, 2025, [[paper](https://openreview.net/pdf?id=TbTJJNjumY)]

&bull; DualOpt: A Dual Divide-and-Optimize Algorithm for the Large-scale Traveling Salesman Problem, ***AAAI***, 2025, [[paper](https://arxiv.org/abs/2501.08565)]

&bull; Destroy and Repair Using Hyper-Graphs for Routing, ***AAAI***, 2025, [[paper](https://www.arxiv.org/pdf/2502.16170)]

## NCO Solvers for Solving Multi-constrained VRP Variants

&bull; Efficient Training of Multi-task Combinatorial Neural Solver with Multi-armed Bandits, ***arXiv***, 2023, [[paper](https://arxiv.org/pdf/2305.06361)]

&bull; A Neural Column Generation Approach to the Vehicle Routing Problem with Two-Dimensional Loading and Last-In-First-Out Constraints, ***IJCAI***, 2024, [[paper](https://www.ijcai.org/proceedings/2024/0218.pdf)]

&bull; CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention, ***arXiv***, 2024, [[paper](https://arxiv.org/abs/2412.00346)]

&bull; Cross-problem learning for solving vehicle routing problems, ***IJCAI***, 2024, [[paper](https://arxiv.org/pdf/2404.11677)]

&bull; Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization, ***SIGKDD***, 2024, [[paper](https://arxiv.org/pdf/2402.16891)]

&bull; MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts, ***ICML***, 2024, [[paper](https://arxiv.org/pdf/2405.01029)]

&bull; Prompt Learning for Generalized Vehicle Routing, ***IJCAI***, 2024, [[paper](https://arxiv.org/pdf/2405.12262)]

&bull; RouteFinder: Towards Foundation Models for Vehicle Routing Problems, ***ICML workshop***, 2024, [[paper](https://openreview.net/forum?id=hCiaiZ6e4G)]

&bull; UNCO: Towards Unifying Neural Combinatorial Optimization through Large Language Model, ***arXiv***, 2024, [[paper](https://arxiv.org/abs/2408.12214)]

&bull; Goal: A generalist combinatorial optimization agent learner, ***ICLR***, 2025 [[paper](https://openreview.net/forum?id=z2z9suDRjw)]

# Acknowledgements 📜
This is an open collaborative research project among:

<a href="https://ccst.jlu.edu.cn/">
    <img src="https://github.com/wuuu110/NCO-solvers-for-VRP/blob/main/img-folder/jlu.jpg" height = 50/>
</a>
<a href="https://www.ntu.edu.sg/lily">
    <img src="https://github.com/wuuu110/NCO-solvers-for-VRP/blob/main/img-folder/ntu.svg" height = 50/>
</a>
<a href="https://www.thss.tsinghua.edu.cn/">
    <img src="https://github.com/wuuu110/NCO-solvers-for-VRP/blob/main/img-folder/Thu.svg" height = 50/>
</a>



