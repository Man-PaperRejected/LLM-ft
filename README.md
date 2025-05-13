# 🤖 大语言模型下游任务微调实践

本仓库专注于对大语言模型（LLMs）在各种自然语言处理（NLP）下游任务上进行微调的**代码实践**。主要目标是通过使用像 `transformers`、`peft` 这样的流行库，以及将强大的预训练模型应用于特定问题的整个工作流程，积累实践经验。


## 🎯 项目目的

本仓库的主要目标是**熟悉大语言模型微调涉及的代码**。重点不在于取得最先进的成果，而在于：
* 理解微调过程的各个步骤。
* 使用 `transformers` 库的 API。
* 探索不同的微调技术（如 LoRA/PEFT）。
* 实现数据加载和预处理流程。
* 进行实验并解读基本结果。

## 📂 仓库结构

代码库按照下游任务组织在 `finetune/` 目录下：
```
.
└── finetune
    ├── QA
    │   ├── GPT2 - Race.ipynb
    │   ├── results
    │   └── RoBERTa - Race.ipynb
    ├── Seq - Cls
    │   ├── bert - imdb.ipynb
    │   ├── bert - peft.py
    │   ├── bert.py
    │   ├── pipeline.py
    │   └── results
    └── Translation
```
## ✨ 已实现的任务

目前，仓库包含以下实践示例：

| 任务类型 | 数据集 | 模型 | 微调 |
| --- | --- | --- | --- |
| 问答（QA） | RACE  | GPT-2（w/o Task-Head）  | P-tuning v2 |
| 问答（QA） | RACE  | RoBERTa（w Task-Head）  | Lora |
| 序列分类 | IMDB | BERT | 部分参数解冻，Lora |

## 🚀 快速开始

### 环境
* Python 3.10
* Pytorch 2.5.1
* transformer 4.49.0

其余环境设置已经导出到 env.yaml

1. 克隆仓库：
    ```bash
    git clone https://github.com/Man-PaperRejected/LLM-ft.git
    cd YOUR_REPO_NAME
    ```
2. （推荐）创建虚拟环境（虽然好像成功率蛮低的）：
    ```bash
    conda env create -f env.yml
    conda activate llm
    ```
    也可以在已有环境上安装相关库， 见requirements.txt
3. 安装所需的库。
    ```bash
    pip install -r requirements.txt
    ```

### 运行方法
1. 导航到你感兴趣的特定任务目录，例如：
    ```bash
    cd finetune/Seq - Cls
    ```
2. 运行 Python 脚本或 Jupyter Notebook：
    * 对于 `.py` 脚本：
        ```bash
        python bert.py
        ```
    * 对于 `.ipynb` 笔记本：
        ```bash
        jupyter lab # 或者 jupyter notebook
        ```
        然后从文件浏览器中打开所需的笔记本。

*(注意：运行这些示例可能会使用 `transformers` 和 `datasets` 库自动下载预训练模型和数据集。请确保你有足够的磁盘空间和稳定的网络连接。，下载慢可以参考一下这个[博客](https://bbs.huaweicloud.com/blogs/432514))*

## 📝 结果
运行微调脚本或笔记本所产生的任何输出结果、指标或保存的模型，都将存储在每个任务文件夹内各自的 `results/` 子目录中。

## 👋 贡献
本仓库主要用于个人学习和实践。不过，欢迎通过提交 Issue 或 Pull Request 的方式提出改进建议、修复 bug 或提供替代实现方案。


## 🙏 致谢
* 感谢 `transformers`、`datasets` 和 `peft` 库的开发者。
* 感谢示例中使用的公共数据集的提供者。

---

#### Happy Coding! ✨
