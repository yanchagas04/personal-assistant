"""Módulo de Fine-Tuning e Transfer Learning com Unsloth (LoRA / QLoRA).

Projetado para treinar modelos da família LLaMA 3 / LLaMA 3.2 com aceleração de GPU
e exportar o modelo ajustado diretamente para o formato GGUF (compatível com Ollama).
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
import typer

from personal_assistant.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(
    help="CLI para realizar Fine-Tuning de LLMs com Unsloth (LoRA) e exportação para GGUF."
)


def load_dialogues(dataset_path: Path) -> list[dict[str, Any]]:
    """Carrega conversas estruturadas em formato JSONL."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {dataset_path}")

    dialogues: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))

    return dialogues


def prepare_hf_dataset(dialogues: list[dict[str, Any]], tokenizer: Any) -> Any:
    """Aplica o template de chat do LLaMA e converte a lista em um Dataset Hugging Face."""
    from datasets import Dataset

    def format_convo(examples: dict[str, Any]) -> dict[str, list[str]]:
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = Dataset.from_list(dialogues)
    return dataset.map(format_convo, batched=True)


def setup_model_and_lora(
    model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
) -> tuple[Any, Any]:
    """Inicializa o modelo base quantizado em 4-bit e aplica adaptadores LoRA via Unsloth."""
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError as e:
        logger.error(
            "A biblioteca 'unsloth' não está instalada neste ambiente.\n"
            "O Unsloth requer ambiente Linux com GPU (ex: Google Colab com GPU T4).\n"
            "Instale com: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )
        raise typer.Exit(code=1) from e

    logger.info(f"Carregando modelo base 4-bit: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    logger.info(f"Aplicando adaptadores LoRA (r={lora_r}, alpha={lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer


def train_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    output_dir: Path,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
) -> Any:
    """Executa o treinamento supervisionado (SFT) com SFTTrainer."""
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported

    logger.info(f"Iniciando treinamento SFT ({epochs} épocas, lr={learning_rate})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=str(output_dir),
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()
    logger.success("Treinamento finalizado com sucesso!")
    return trainer_stats


def export_to_gguf(
    model: Any,
    tokenizer: Any,
    gguf_output_dir: Path,
    quantization_method: str = "q4_k_m",
) -> None:
    """Exporta o modelo treinado diretamente para o formato GGUF para importação no Ollama."""
    logger.info(f"Exportando modelo para GGUF (quantização: {quantization_method})...")
    gguf_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf_output_dir),
        tokenizer,
        quantization_method=quantization_method,
    )
    logger.success(f"Modelo GGUF exportado em: {gguf_output_dir}")


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "personal-answers.jsonl",
    model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    output_dir: Path = MODELS_DIR / "yan_clone_unsloth",
    gguf_output_dir: Path = MODELS_DIR / "yan_clone_gguf",
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    export_gguf: bool = True,
    quantization_method: str = "q4_k_m",
) -> None:
    """Pipeline de Fine-Tuning com Unsloth (LoRA/QLoRA) e exportação GGUF para Ollama."""
    logger.info(f"Carregando dados de: {dataset_path}")
    dialogues = load_dialogues(dataset_path)
    logger.info(f"{len(dialogues)} diálogos carregados.")

    model, tokenizer = setup_model_and_lora(model_name=model_name)
    dataset = prepare_hf_dataset(dialogues, tokenizer)

    train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    if export_gguf:
        export_to_gguf(
            model=model,
            tokenizer=tokenizer,
            gguf_output_dir=gguf_output_dir,
            quantization_method=quantization_method,
        )


if __name__ == "__main__":
    app()
