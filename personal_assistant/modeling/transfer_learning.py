"""Módulo de Transfer Learning e Customização de Modelos com a biblioteca Ollama.

Permite gerar Modelfiles com aprendizado em contexto (few-shot), definir hiperparâmetros,
instruções de sistema para o clone digital de Yan Chagas e compilar o modelo customizado no Ollama.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
import typer

from personal_assistant.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(
    help="CLI para realizar Transfer Learning e criação de modelo customizado com Ollama."
)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Carrega o dataset de conversas a partir de um arquivo JSONL."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Arquivo de dataset não encontrado em: {dataset_path}")

    dialogues: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))

    return dialogues


def build_modelfile_content(
    dialogues: list[dict[str, Any]],
    base_model: str = "llama3.2",
    clone_name: str = "Yan Chagas",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_examples: int = 15,
) -> str:
    """Gera o conteúdo textual de um Modelfile para o Ollama com in-context few-shot learning."""
    system_prompt = (
        f"Você é o clone digital autêntico de {clone_name}. "
        "Responda sempre direto ao ponto, de forma descontraída, autêntica e informal, "
        "exatamente como nas conversas reais com seus amigos. "
        "Use linguagem natural brasileira com gírias espontâneas (ex: 'zorra', 'massa', 'tranquilo', 'tô na correria'), "
        "evite explicações prolixas e nunca aja como um assistente corporativo ou robótico."
    )

    lines: list[str] = [
        f"FROM {base_model}",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        'PARAMETER stop "<|eot_id|>"',
        'PARAMETER stop "<|end_of_text|>"',
        "",
        f'SYSTEM """{system_prompt}"""',
        "",
        "# ========================================================",
        "# Exemplos de Transfer Learning (In-Context Few-Shot)",
        "# ========================================================",
    ]

    included = 0
    for dialogue in dialogues:
        messages = dialogue.get("messages", [])
        for msg in messages:
            role = msg.get("role")
            if role in ["user", "assistant"]:
                sanitized_content = msg.get("content", "").replace('"', '\\"')
                lines.append(f'MESSAGE {role} "{sanitized_content}"')
        included += 1
        if included >= max_examples:
            break

    return "\n".join(lines)


def save_modelfile(content: str, output_path: Path) -> Path:
    """Salva o Modelfile gerado no sistema de arquivos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def register_ollama_model(model_name: str, modelfile_content: str) -> bool:
    """Registra e cria o modelo customizado no servidor Ollama local via API Python."""
    try:
        import ollama

        logger.info(f"Registrando modelo '{model_name}' no Ollama via API Python...")
        progress_stream = ollama.create(model=model_name, modelfile=modelfile_content, stream=True)
        for step in progress_stream:
            status = step.get("status", "")
            if status:
                logger.debug(f"Ollama status: {status}")

        logger.success(f"Modelo '{model_name}' criado com sucesso no Ollama!")
        return True
    except Exception as e:
        logger.warning(f"Não foi possível criar o modelo via API do Ollama: {e}")
        logger.info(
            "Certifique-se de que o Ollama está rodando ou use: ollama create <nome> -f <Modelfile>"
        )
        return False


def test_model_inference(model_name: str, prompt: str) -> str:
    """Envia um prompt para o modelo no Ollama e retorna a resposta gerada."""
    try:
        import ollama

        logger.info(f"Testando inferência com prompt: '{prompt}'...")
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.get("message", {}).get("content", "")
    except Exception as e:
        return f"[Erro ao executar inferência: {e}]"


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "personal-answers.jsonl",
    modelfile_path: Path = MODELS_DIR / "Modelfile.yan_clone",
    base_model: str = "llama3.2",
    model_name: str = "yan-clone",
    clone_name: str = "Yan Chagas",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_examples: int = 15,
    run_inference: bool = True,
    test_prompt: str = "Bora almoçar onde hoje?",
) -> None:
    """Pipeline de Transfer Learning via Modelfile e Ollama."""
    logger.info(f"Carregando dataset de: {dataset_path}")
    dialogues = load_dataset(dataset_path)
    logger.info(f"{len(dialogues)} conversas carregadas para injeção de estilo.")

    logger.info(f"Construindo Modelfile com base no modelo '{base_model}'...")
    modelfile_content = build_modelfile_content(
        dialogues=dialogues,
        base_model=base_model,
        clone_name=clone_name,
        temperature=temperature,
        top_p=top_p,
        max_examples=max_examples,
    )

    save_modelfile(modelfile_content, modelfile_path)
    logger.success(f"Modelfile salvo com sucesso em: {modelfile_path}")

    created = register_ollama_model(model_name=model_name, modelfile_content=modelfile_content)

    if created and run_inference:
        answer = test_model_inference(model_name=model_name, prompt=test_prompt)
        logger.info(f"Resposta do clone:\n{answer}")


if __name__ == "__main__":
    app()
