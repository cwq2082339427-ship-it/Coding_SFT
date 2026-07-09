"""
Qwen2.5-4B SFT 训练脚本
================================

流程:
  1. 使用 myDataset 清洗管线处理原始数据 → output/cleaned_data.jsonl
  2. 加载 Qwen2.5-4B-Instruct 模型 + LoRA
  3. SFT 训练
  4. 保存 LoRA adapter
  5. 推理效果测试

使用方法:
    python train.py                          # 默认训练
    python train.py --model_id Qwen/Qwen2.5-7B-Instruct  # 换模型
    python train.py --resume                 # 从 checkpoint 恢复
    python train.py --test_only              # 仅测试已训练的模型
    python train.py --merge                  # 训练后合并 LoRA 权重

依赖:
    pip install torch transformers datasets trl peft accelerate
    pip install sentence-transformers faiss-cpu   # 用于数据清洗
"""

import os
import sys
import json
import math
import logging
import argparse
from typing import Optional, Dict, List, Tuple
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("sft_trainer")


class TrainingConfig:

    model_name_or_path: str = "Qwen/Qwen2.5-4B-Instruct"

    cleaned_data_path: str = os.path.join(
        os.path.dirname(__file__), "output", "cleaned_data.jsonl"
    )
    max_seq_length: int = 2048
    eval_ratio: float = 0.01

    # ---- LoRA ----
    lora_r: int = 32
    """LoRA alpha 缩放系数，通常为 r 的 2 倍"""
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    """检查点和最终模型保存目录"""
    output_dir: str = os.path.join(
        os.path.dirname(__file__), "checkpoints"
    )
    num_train_epochs: int = 3
    per_device_batch_size: int = 4
    """梯度累积步数（有效 batch = per_device_batch_size × gradient_accumulation_steps）"""
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"

    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()


    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    """梯度检查点（节省显存，略微降低速度）"""
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    report_to: str = "none"
    resume_from_checkpoint: bool = False
    use_semantic_dedup: bool = True

    test_prompt: str = "用 Python 实现一个快速排序算法。"



class ProgressCallback(TrainerCallback):

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, **kwargs):
        if state.is_local_process_zero and state.log_history:
            last_log = state.log_history[-1]
            loss = last_log.get("loss", "N/A")
            lr = last_log.get("learning_rate", "N/A")
            epoch = last_log.get("epoch", 0)
            logger.info(
                f"Step {state.global_step:>6d} | "
                f"Epoch {epoch:.2f} | "
                f"Loss {loss if isinstance(loss, str) else f'{loss:.4f}'} | "
                f"LR {lr if isinstance(lr, str) else f'{lr:.2e}'}"
            )


def load_cleaned_data(
    filepath: str,
    use_semantic_dedup: bool = True,
) -> "Dataset":

    if not os.path.exists(filepath):
        logger.warning(f"清洗数据文件不存在: {filepath}")
        logger.info("自动运行清洗管线生成数据...")

        # 动态导入清洗模块
        sys.path.insert(0, os.path.dirname(__file__))
        from myDataset.clean_basic_length import run_pipeline, save_jsonl

        clean_data = run_pipeline(use_semantic_dedup=use_semantic_dedup)
        save_jsonl(clean_data, filepath)
        logger.info(
            f"清洗完成: {len(clean_data)} 条样本，已保存至 {filepath}"
        )

    dataset = load_dataset("json", data_files=filepath, split="train")
    logger.info(f"加载清洗后数据: {len(dataset)} 条样本")
    return dataset


# =====================================================================
# 模型 & Tokenizer 加载
# =====================================================================

def load_model_and_tokenizer(
    model_name_or_path: str,
    cfg: TrainingConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载 Qwen2.5 模型和 tokenizer。

    注意:
      - Qwen2.5 默认没有 pad_token，我们复用 eos_token。
      - 训练时关闭 kv cache（model.config.use_cache = False）。

    参数:
        model_name_or_path: HuggingFace 模型名称或本地路径
        cfg: 训练配置，用于决定精度 (bf16/fp16)
    """
    logger.info(f"正在加载模型: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    # Qwen2.5 没有显式设置 pad_token => 用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"pad_token 未设置，已复用 eos_token: {tokenizer.eos_token}")

    # 检查 chat_template
    if tokenizer.chat_template is None:
        logger.warning("Tokenizer 没有 chat_template，将使用默认格式")
    else:
        logger.info("Tokenizer 已内置 chat_template")

    torch_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # 训练时必须关闭 use_cache 以配合 gradient checkpointing
    model.config.use_cache = False

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"模型参数量: {total_params / 1e9:.2f}B | "
        f"可训练: {trainable_params / 1e6:.2f}M"
    )

    return model, tokenizer


# =====================================================================
# LoRA 配置
# =====================================================================

def setup_lora(
    model: AutoModelForCausalLM,
    config: TrainingConfig,
) -> PeftModel:

    if config.lora_target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        logger.info(f"使用默认 LoRA 目标模块: {target_modules}")
    else:
        target_modules = config.lora_target_modules
        logger.info(f"使用自定义 LoRA 目标模块: {target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(config: TrainingConfig):
    """执行完整的 SFT 训练流程"""

    # ---- 1. 加载数据 ----
    logger.info("=" * 60)
    logger.info("步骤 1/5: 加载清洗后数据")
    logger.info("=" * 60)

    dataset = load_cleaned_data(config.cleaned_data_path, config.use_semantic_dedup)

    # 划分训练/验证集
    split_dataset = dataset.train_test_split(
        test_size=config.eval_ratio,
        seed=42,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(
        f"训练集: {len(train_dataset):>6d} 条 | "
        f"验证集: {len(eval_dataset):>6d} 条"
    )

    # ---- 2. 加载模型 ----
    logger.info("=" * 60)
    logger.info("步骤 2/5: 加载模型和 Tokenizer")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(config.model_name_or_path, config)

    # 数据集格式化：将 messages 转为模型所需的 text 格式
    logger.info("正在将数据集格式化为对话文本...")

    def format_chat(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    train_dataset = train_dataset.map(format_chat)
    eval_dataset = eval_dataset.map(format_chat)
    logger.info("数据集格式化完成")

    # 打印一条样例供检查
    logger.info(f"格式化样例:\n{train_dataset[0]['text'][:300]}...")

    # ---- 3. 配置 LoRA ----
    logger.info("=" * 60)
    logger.info("步骤 3/5: 配置 LoRA")
    logger.info("=" * 60)

    model = setup_lora(model, config)

    # 梯度检查点兼容性处理
    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    # ---- 4. 训练参数 ----
    logger.info("=" * 60)
    logger.info("步骤 4/5: 配置训练参数")
    logger.info("=" * 60)

    # 有效 batch size 信息
    effective_batch = (
        config.per_device_batch_size
        * config.gradient_accumulation_steps
        * max(1, torch.cuda.device_count())
    )
    logger.info(f"每设备 batch size:      {config.per_device_batch_size}")
    logger.info(f"梯度累积步数:           {config.gradient_accumulation_steps}")
    logger.info(f"有效 batch size:        {effective_batch}")
    logger.info(f"学习率:                 {config.learning_rate:.2e}")
    logger.info(f"训练轮数:               {config.num_train_epochs}")
    logger.info(f"最大序列长度:           {config.max_seq_length}")
    logger.info(f"精度:                   {'bf16' if config.bf16 else 'fp16'}")
    logger.info(f"输出目录:               {config.output_dir}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        # 训练策略
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        # 学习率
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        # 精度
        bf16=config.bf16,
        fp16=config.fp16,
        # 日志 & 保存
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 报告
        report_to=config.report_to,
        # 数据加载
        dataloader_num_workers=config.dataloader_num_workers,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        # 断点续训
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    # ---- 5. 创建 Trainer 并训练 ----
    logger.info("=" * 60)
    logger.info("步骤 5/5: 开始 SFT 训练")
    logger.info("=" * 60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        neftune_noise_alpha=config.neftune_noise_alpha,
        callbacks=[ProgressCallback()],
    )

    # 检查是否有可恢复的 checkpoint
    last_checkpoint = None
    if config.resume_from_checkpoint and os.path.isdir(config.output_dir):
        checkpoints = [
            d for d in os.listdir(config.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # 按 checkpoint 编号排序，取最新的
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(config.output_dir, checkpoints[-1])
            logger.info(f"发现已有 checkpoint，将从该点恢复: {last_checkpoint}")

    # 开始训练
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- 6. 保存最终模型 ----
    logger.info("=" * 60)
    logger.info("训练完成！保存最终模型...")
    logger.info("=" * 60)

    final_adapter_dir = os.path.join(config.output_dir, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)

    # 保存 LoRA adapter
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # 同时保存训练配置供后续推理使用
    config_path = os.path.join(final_adapter_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(config), ensure_ascii=False, indent=2))

    logger.info(f"LoRA adapter 已保存至: {final_adapter_dir}")
    logger.info("训练全部结束！")


# =====================================================================
# 推理测试
# =====================================================================

@torch.no_grad()
def test_inference(
    config: TrainingConfig,
    adapter_path: Optional[str] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    加载训练好的 LoRA adapter 进行推理测试。

    参数:
        adapter_path: LoRA adapter 路径（默认使用 output_dir/final_adapter）
        prompt: 测试用 prompt
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        top_p: 核采样阈值
    返回:
        模型生成的文本
    """
    if adapter_path is None:
        adapter_path = os.path.join(config.output_dir, "final_adapter")
    if prompt is None:
        prompt = config.test_prompt

    if not os.path.exists(adapter_path):
        logger.error(f"Adapter 路径不存在: {adapter_path}")
        logger.error("请先运行 python train.py 进行训练")
        return ""

    logger.info(f"加载 adapter 进行推理测试: {adapter_path}")
    logger.info(f"测试 prompt: {prompt}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 base model
    torch_dtype = torch.bfloat16 if config.bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model.config.use_cache = True  # 推理时开启 kv cache 加速

    # 格式化 prompt
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    logger.info(f"模型回复:\n{'-' * 40}\n{response}\n{'-' * 40}")
    return response


# =====================================================================
# 合并 LoRA 权重到 Base Model
# =====================================================================

def merge_and_save(config: TrainingConfig):
    """
    将 LoRA adapter 合并到 base model 并保存完整模型。

    合并后的模型不再依赖 peft，可直接用 AutoModelForCausalLM 加载，
    推理速度更快，且可用于部署。
    """
    adapter_path = os.path.join(config.output_dir, "final_adapter")
    merge_path = os.path.join(config.output_dir, "merged_model")

    if not os.path.exists(adapter_path):
        logger.error(f"Adapter 路径不存在: {adapter_path}")
        return

    logger.info(f"加载 base model: {config.model_name_or_path}")
    logger.info(f"加载 LoRA adapter: {adapter_path}")

    torch_dtype = torch.bfloat16 if config.bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info("正在合并 LoRA 权重到 base model...")
    merged_model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )

    os.makedirs(merge_path, exist_ok=True)
    merged_model.save_pretrained(merge_path)
    tokenizer.save_pretrained(merge_path)

    logger.info(f"合并后的完整模型已保存至: {merge_path}")
    logger.info(f"你可以使用以下代码加载合并后的模型:")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{merge_path}')")


# =====================================================================
# 主入口
# =====================================================================

def parse_args() -> argparse.Namespace:
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="Qwen2.5 SFT 训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="模型名称或路径（覆盖 config 中的默认值）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="每设备 batch size",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="学习率",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=None,
        help="最大序列长度",
    )
    parser.add_argument(
        "--lora_r", type=int, default=None,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=None,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="从最近的 checkpoint 恢复训练",
    )
    parser.add_argument(
        "--no_semantic_dedup", action="store_true", default=False,
        help="运行清洗管线时不使用语义去重（可加速首次运行）",
    )
    parser.add_argument(
        "--test_only", action="store_true", default=False,
        help="仅运行推理测试，不进行训练",
    )
    parser.add_argument(
        "--merge", action="store_true", default=False,
        help="训练后将 LoRA 权重合并到 base model 并保存完整模型",
    )
    parser.add_argument(
        "--report_to", type=str, default=None,
        choices=["none", "wandb", "tensorboard"],
        help="日志报告目标",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="清洗后数据的 JSONL 路径",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    global config
    config = TrainingConfig()

    # 命令行参数覆盖默认配置
    if args.model_id:
        config.model_name_or_path = args.model_id
    if args.batch_size:
        config.per_device_batch_size = args.batch_size
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.lora_alpha:
        config.lora_alpha = args.lora_alpha
    if args.resume:
        config.resume_from_checkpoint = True
    if args.report_to:
        config.report_to = args.report_to
    if args.data_path:
        config.cleaned_data_path = args.data_path
    if args.no_semantic_dedup:
        config.use_semantic_dedup = False

    logger.info("=" * 60)
    logger.info(f"模型: {config.model_name_or_path}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"数据路径: {config.cleaned_data_path}")
    logger.info(f"语义去重: {'关闭' if not config.use_semantic_dedup else '开启'}")
    logger.info("=" * 60)

    # 模式分发
    if args.test_only:
        test_inference(config)
    elif args.merge:
        merge_and_save(config)
    else:
        train(config)
        # 训练结束时自动做一次推理测试
        logger.info("\n训练完成，运行推理测试...")
        test_inference(config)


if __name__ == "__main__":
    main()
