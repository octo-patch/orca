<div align="center">
<img src="./assets/orca-logo.png" width="340"/>
</div>

<h1 align="center">Orca: The World is in Your Mind</h1>

<p align="center">
  <b>Orca Team, Beijing Academy of Artificial Intelligence</b>
</p>

<p align="center">
  ⭐️ <a href="https://orca-wm.github.io">Project Page</a>
  &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/papers/2606.30534">Hugging Face</a>
  &nbsp;|&nbsp;
  📑 <a href="https://arxiv.org/abs/2606.30534">Technical Report</a>
</p>

<p align="center">
  <b>A general world foundation model centered on Next-State-Prediction.</b>
</p>

<p align="center">
  💬 <b>If you have any questions, feel free to contact us via WeChat.</b>
</p>

<div align="center">
<img src="./assets/wechat.jpg" width="620"/>
</div>

## 🔥 Overview

**Orca** is an initial instantiation of a general world foundation model. It learns a unified world latent space from multimodal world signals and exposes the learned latent through multimodal readout interfaces.

Rather than optimizing isolated **next-token**, **next-frame**, or **next-action** prediction objectives, Orca is centered on **Next-State-Prediction**: a unified state-transition modeling route toward understanding, predicting, and acting upon the world. In this version, Orca focuses on two fundamental input signals: **visual signals** for dense observations of world evolution, and **language signals** for event descriptions, task intentions, causal explanations, and semantic constraints.

- **Unconscious learning**: dense natural transitions from continuous videos.
- **Conscious learning**: sparse meaningful transitions under language-described events and VQA supervision.
- **Frozen-backbone readouts**: lightweight decoders for **text**, **images**, and **actions**.
- **Scaling analysis**: stronger world modeling, stronger downstream readouts.

## 🗞️ News

- **`2026-06-29`**: 🎉 [**Orca Technical Report**](https://arxiv.org/abs/2606.30534) was released.

## 📆 Todo

- [x] Release the **Orca Technical Report**.
- [ ] Release the **Orca-4B checkpoint** for world latent learning and downstream readouts.
- [ ] Release the **Orca-0.8B checkpoint** for lightweight research and reproduction.
- [ ] Release **inference code** for text, image, and action readouts.
- [ ] Release **downstream fine-tuning code** for modality-specific readout adaptation.

## ⭐️ Architecture

Orca follows an **Encoder-Decoder** architecture. Given multimodal world signals, the **Encoder** learns a world latent through unconscious and conscious learning. After pre-training, the Encoder is frozen, and only lightweight modality-specific decoders are trained to read out the latent into downstream modalities.

<div align="center">
<img src="./assets/teaser.png" width="850"/>
</div>

<p align="center"><b>Figure 1.</b> Orca learns a unified world latent through unconscious and conscious learning.</p>

<br/>

<div align="center">
<img src="./assets/readout.png" width="850"/>
</div>

<p align="center"><b>Figure 2.</b> Lightweight readouts adapt frozen world latents to language, vision, and action.</p>

Orca models world-state transitions under both **implicit dynamics** and **explicit conditions**. Implicit dynamics capture latent or unobserved factors such as physical laws, object properties, scene dynamics, and environmental forces, while explicit conditions describe observed signals such as human instructions, event descriptions, task intentions, or causal premises.

## 📚 Data

For pre-training, Orca constructs a large-scale world-learning inventory from **visual signals** and **language signals**. The data mixture includes video data for observation-only state transitions, event data for event-conditioned state transitions, and VQA data for response generation.

- **125K hours** of video data covering egocentric interaction, exocentric manipulation, robot execution, and natural dynamics.
- **160M** event annotations with fine- and coarse-grained captions for event-level transition learning.
- **General VQA data** for aligning world latents with language understanding and response generation.

<div align="center">
<img src="./assets/datapipeline.png" width="850"/>
</div>

<p align="center"><b>Figure 3.</b> Orca data pipeline from multimodal world signals to world latent learning.</p>

## 🔍 Evaluation

Orca is evaluated through three representative downstream readouts:

- **Text generation**: understanding on *TemporalBench*, *MVBench*, *SWITCH*, and *3DSRBench*.
- **Image prediction**: future-state prediction on *PRICE-V0.1* real-world interactions.
- **Action generation**: five real-robot tasks under *environment* and *object OOD* settings.

<div align="center">
<img src="./assets/orca-scaling-performance.png" width="850"/>
</div>

Experiments indicate that stronger world latents from pre-training lead to stronger downstream readouts. As pre-training scales up, Orca improves across text, image, and action readouts while keeping the backbone frozen during readout post-training.

## 🤗 Model Zoo

Model links will be added after release.

| Model | Checkpoint | Description |
| --- | --- | --- |
| Orca-0.8B | Coming soon | Lightweight Orca backbone for world latent learning. |
| Orca-4B | Coming soon | Larger Orca backbone with stronger downstream readout performance. |

## 🛠️ Usage

Code, checkpoints, and inference examples will be released soon.

## 📑 Citation

If you find Orca useful for your research, please consider citing our technical report.

```bibtex
@article{orca2026,
  title={Orca: The World is in Your Mind},
  author={Yihao Wang and Yuheng Ji and Mingyu Cao and Yanqing Shen and Runze Xiao and Huaihai Lyu and Senwei Xie and Euan Liu and Klara Tian and Tianfeng Long and Yichi Zhang and Zhengliang Cai and Ruike Chen and Jifan Zhao and Ruochuan Shi and Zihan Tang and Jing Lyu and Wenxing Tan and Ningbo Zhang and Yangtao Hu and Yuming Gao and Xiansheng Chen and Junkai Zhao and Congsheng Xu and Boan Zhu and Ziqi Wang and Yupu Feng and Qiongqiong Zhang and Yingli Zhao and Yulong Ao and Shaoxuan Xie and You Liu and Guocai Yao and Leiduo Zhang and Xiaodan Liu and Yunyan Zhang and Yance Jiao and Xinyan Yang and Jiaxing Wei and Xu Liu and Tengfei Pan and Shaokai Nie and Chunlei Men and Sen Cui and Xiaojie Jin and Hongyang Li and Jianlan Luo and Yao Mu and Yunchao Wei and Jun Yan and Hang Zhao and Xiaolong Zheng and Jiaming Li and Yonghua Lin and Tiejun Huang and Zhongyuan Wang and Pengwei Wang},
  journal={arXiv preprint arXiv:2606.30534},
  year={2026}
}
```
