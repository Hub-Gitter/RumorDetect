# 可解释谣言检测系统

基于 Twitter-RoBERTa + DeepSeek V3.2 的谣言检测系统。输入一条英文推文，输出：(1) 谣言/非谣言分类，(2) 自然语言解释。

## 环境要求

- Python 3.10+
- PyTorch 2.x（CUDA 可选，CPU 也可运行）
- GPU 建议 >= 6GB VRAM（LoRA 微调仅需 ~3GB）

## 安装

```bash
git clone git@github.com:Hub-Gitter/rumer2026.git
cd rumer2026
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"
```

## 配置 API Key

编辑 `.env` 文件，将占位内容替换为你的 API Key：

```
SJTU_API_KEY=your-api-key
```

默认使用 OpenAI 兼容接口，替换为其他 API 时修改 `src/llm_explain.py` 中的 `base_url` 和 `model` 即可。

## 模型权重

默认基模为 `cardiffnlp/twitter-roberta-base`，使用 LoRA 微调，adapter
训练后保存在 `checkpoints/lora_adapter/`。

基础模型由 HuggingFace 自动下载缓存，无需手动下载。若当前 checkpoint 是旧
`roberta-base` 训练得到的，推理脚本会拒绝加载并提示重新训练，避免把旧 adapter
错误套到新基模上。

## 使用方法

完成 Twitter-RoBERTa 重训并生成匹配的 `checkpoints/` 后，可运行检测：

### 单条推文检测

```bash
python src/main.py --text "NASA confirms successful launch of new Mars rover"
```

输出示例：
```
============================================================
Input:      NASA confirms successful launch of new Mars rover
Prediction: NON-RUMOR (confidence: 96.9%)
Key tokens: ('confirms', 0.05), ('launch', 0.07), ('NASA', 0.04)
Explanation: "confirms" indicates an official announcement, and "NASA"
  names a specific authoritative source. The factual, measured tone
  with no emotional or speculative language supports credibility.
============================================================
```

### 批量检测

```bash
# 标准批量（分类 + 逐条 LLM 解释，约 40 分钟）
python src/main.py --batch data/val.csv --output outputs/

# 两阶段批量（分类快 + 批量 LLM 调用，约 13 分钟，推荐）
python src/batch_pipeline.py
```

结果保存到 `outputs/predictions.csv` 和 `outputs/explanations.csv`。

`predictions.csv` 会保留输入文件中的 `id`、`label`、`event`（如存在），并追加
`label_pred`、`confidence`、`key_tokens`、`explanation`。因此可直接用于总体指标和
按事件指标评测。

## 评测

```bash
# 对预测结果计算准确率、F1、混淆矩阵等
python src/evaluate.py --predictions outputs/predictions.csv --ground_truth data/val.csv

# 仅运行分类（不含 LLM），用于快速评测模型
python src/predict_batch.py --input data/val.csv --output outputs/predictions.csv --checkpoint-dir checkpoints
```

若预测文件已包含 `label` 列，可省略 `--ground_truth`。

## 训练（可选）

```bash
# 使用 Twitter-RoBERTa + Focal Loss 训练（默认）
python src/train_final.py

# 启用 R-Drop 正则化
python src/train_final.py --rdrop

# 回退到标准交叉熵损失
python src/train_final.py --no-focal-loss

# Leave-One-Event-Out 交叉验证（7 轮）
python src/train_loeo.py
```

训练参数：LoRA r=8, alpha=16, lr=5e-4, batch_size=16, max_epochs=5, Focal Loss γ=2.0, early_stopping patience=2。

## 项目结构

```
rumer2026/
├── README.md
├── report.pdf                 # 大作业报告
├── requirements.txt
├── .env.example               # API Key 配置模板
├── data/
│   ├── train.csv              # 训练集 (2840条)
│   └── val.csv                # 验证集 (401条)
├── src/
│   ├── preprocess.py          # 文本预处理 (URL/@/# 规范化)
│   ├── dataset.py             # PyTorch Dataset
│   ├── model.py               # Twitter-RoBERTa + LoRA 分类器
│   ├── train_loeo.py          # LOEO 7 轮训练
│   ├── train_final.py         # 全量数据最终模型训练
│   ├── data_augment.py        # 数据增强 (同义词替换)
│   ├── explain_tokens.py      # Gradient×Input 关键词提取
│   ├── llm_explain.py         # DeepSeek V3.2 解释生成
│   ├── evaluate.py            # 评测脚本
│   ├── eval_explanations.py   # 解释质量量化评估
│   ├── main.py                # 端到端入口 (单条+批量)
│   ├── batch_pipeline.py      # 两阶段批量流水线
│   └── predict_batch.py       # 纯分类批量推理（不含 LLM）
└── checkpoints/
    ├── lora_adapter/          # LoRA adapter 权重 (~1.2MB)
    └── classifier.pt          # 分类头权重
```

## 当前结果

默认 checkpoint（Focal Loss γ=2.0 训练），在 `data/val.csv` 上的纯分类指标：

| 指标 | val.csv |
|---|---|
| Accuracy | **88.03%** |
| Macro F1 | **0.8789** |
| F1 (非谣言/谣言) | 0.8919 / 0.8659 |
| AUC | 0.9391 |
| 解释引用短语 | 平均 2.72 个/条 |
| LOEO 平均 F1 | 0.5814 |

Focal Loss（γ=2.0）将 F1 从 CE Loss 的 0.8619 提升至 0.8789。R-Drop 通过 `--rdrop` 可选开启（+0.25% Acc，训练时间翻倍），默认不启用。EMA 和 FGM 对抗训练经测试均导致指标下降。

## 接口约定

```python
# 预处理
def preprocess(text: str) -> str

# 分类
def predict(model, classifier, tokenizer, text: str, device: str) -> dict
# → {"label": int, "confidence": float,
#    "prob_non_rumor": float, "prob_rumor": float}

# 关键词提取（需 classifier 和 device 参数）
def get_key_tokens(model, classifier, tokenizer, text: str,
                   target_label: int, top_k: int = 6,
                   device: str = "cpu") -> list
# → [("token", score), ...]

# 解释生成
def generate_explanation(text: str, prediction_label: int,
                         confidence: float, key_tokens: list) -> str

# 批量解释生成
def batch_generate_explanations(items: list, batch_size: int = 5) -> list
```

## 技术方案

```
推文 text → preprocess() → Twitter-RoBERTa+LoRA 分类器 → 标签+置信度
                                                         ↓
                                              Gradient×Input → 关键token
                                                         ↓
                                              DeepSeek V3.2 → 自然语言解释
```

- 分类器：cardiffnlp/twitter-roberta-base + LoRA 微调，Focal Loss（γ=2.0），仅使用文本特征
- 训练策略：Leave-One-Event-Out (LOEO) 交叉验证，保证跨事件泛化
- 可解释性：Gradient × Input (Saliency) 提取关键 token → DeepSeek V3.2 生成自然语言解释
- 大语言模型：SJTU 提供的 DeepSeek V3.2 (685B)，通过 OpenAI 兼容 API 调用
