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

Orca is evaluated through three representative downstream readouts: **text generation**, **image prediction**, and **action generation**.

### Text Generation

Text generation evaluates understanding on *TemporalBench*, *MVBench*, *SWITCH*, and *3DSRBench*.

<table align="center" width="100%">
  <tr>
    <th nowrap>Model</th><th>Size (B)</th><th>MVBench ↑</th><th>TemporalBench ↑</th><th>3DSRBench ↑</th><th>SWITCH ↑</th><th>Avg. ↑</th>
  </tr>
  <tr><td nowrap>Emu3</td><td align="right">8</td><td align="right">35.2</td><td align="right">9.5</td><td align="right">39.1</td><td align="right">38.0</td><td align="right">30.4</td></tr>
  <tr><td nowrap>Emu3.5</td><td align="right">34</td><td align="right">39.5</td><td align="right">9.5</td><td align="right">31.3</td><td align="right">38.9</td><td align="right">29.8</td></tr>
  <tr><td nowrap>MiniCPM-V-4.6</td><td align="right">2</td><td align="right">41.4</td><td align="right">21.2</td><td align="right">47.7</td><td align="right">41.2</td><td align="right">37.9</td></tr>
  <tr><td nowrap>Qwen3.5</td><td align="right">4</td><td align="right"><b>67.1</b></td><td align="right">25.2</td><td align="right">48.1</td><td align="right">42.8</td><td align="right">46.7</td></tr>
  <tr><td nowrap><b>Orca</b></td><td align="right">0.8</td><td align="right">53.6</td><td align="right">22.6</td><td align="right">43.4</td><td align="right">43.7</td><td align="right">40.8</td></tr>
  <tr><td nowrap><b>Orca</b></td><td align="right">4</td><td align="right">65.3</td><td align="right"><b>34.2</b></td><td align="right"><b>52.1</b></td><td align="right"><b>55.6</b></td><td align="right"><b>51.8</b></td></tr>
</table>

### Image Prediction

Image prediction evaluates future-state prediction on *PRICE-V0.1* real-world interactions.

<table align="center" width="100%">
  <tr>
    <th nowrap>Model</th><th>Size (B)</th><th>Gemini<br/>3.1 Pro ↑</th><th>GPT<br/>5.4 ↑</th><th>Doubao-Seed<br/>2.0 ↑</th><th>Gemma<br/>4-31B ↑</th><th>Avg. ↑</th>
  </tr>
  <tr><td nowrap>OmniGen2</td><td align="right">3+4</td><td align="right">24.6</td><td align="right">46.8</td><td align="right">41.4</td><td align="right">45.5</td><td align="right">39.6±10.2</td></tr>
  <tr><td nowrap>FLUX.1-Kontext</td><td align="right">12</td><td align="right">21.6</td><td align="right">46.9</td><td align="right">42.7</td><td align="right">52.5</td><td align="right">40.9±13.5</td></tr>
  <tr><td nowrap>FLUX.2 [klein]</td><td align="right">4+4</td><td align="right">29.7</td><td align="right">64.6</td><td align="right">60.0</td><td align="right"><b>70.2</b></td><td align="right">56.1±18.1</td></tr>
  <tr><td nowrap><b>Orca</b></td><td align="right">0.8+2</td><td align="right">17.0</td><td align="right">48.5</td><td align="right">46.0</td><td align="right">26.5</td><td align="right">34.5±15.3</td></tr>
  <tr><td nowrap><b>Orca</b></td><td align="right">4+2</td><td align="right"><b>44.0</b></td><td align="right"><b>67.9</b></td><td align="right"><b>61.0</b></td><td align="right">66.3</td><td align="right"><b>59.8±10.9</b></td></tr>
</table>

### Action Generation

Action generation evaluates five real-robot manipulation tasks under *environment* and *object OOD* settings.

<table align="center" width="100%">
  <tr>
    <th nowrap>Model</th><th>Rule ↑</th><th>M25 ↑</th><th>M50 ↑</th><th>SR ↑</th><th>MaxP-F ↑</th><th>FNS ↑</th><th>RBS ↑</th><th>SQS ↑</th>
  </tr>
  <tr><td nowrap>V-JEPA 2.1</td><td align="right">17.0</td><td align="right">27</td><td align="right">7</td><td align="right">0</td><td align="right">17.4</td><td align="right">10.1</td><td align="right">20.5</td><td align="right">0.0</td></tr>
  <tr><td nowrap>Qwen3.5</td><td align="right">10.5</td><td align="right">18</td><td align="right">5</td><td align="right">0</td><td align="right">13.1</td><td align="right">7.6</td><td align="right">11.9</td><td align="right">0.0</td></tr>
  <tr><td nowrap>pi0.5</td><td align="right">29.4</td><td align="right">54</td><td align="right"><b>14</b></td><td align="right">5</td><td align="right">26.5</td><td align="right"><b>15.3</b></td><td align="right">26.7</td><td align="right"><b>3.0</b></td></tr>
  <tr><td nowrap><b>Orca</b></td><td align="right"><b>32.4</b></td><td align="right"><b>55</b></td><td align="right"><b>14</b></td><td align="right"><b>6</b></td><td align="right"><b>27.9</b></td><td align="right">15.1</td><td align="right"><b>30.3</b></td><td align="right">2.9</td></tr>
</table>

<sub>M25/M50: trajectories reaching 25%/50% milestones; SR: success rate; MaxP-F: max process in failed trials; FNS: failure near-success score; RBS: robustness score; SQS: success quality score.</sub>

### Scaling Behavior

<div align="center">
<img src="./assets/orca-scaling-performance.png" width="850"/>
</div>

<p align="center"><b>Figure 4.</b> Downstream readout performance improves as Orca pre-training scales.</p>

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
