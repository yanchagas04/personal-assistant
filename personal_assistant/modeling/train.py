import json
from pathlib import Path
import time

from loguru import logger
import typer

from personal_assistant.config import (
    DEFAULT_BASE_MODEL,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    get_gemini_api_key,
)

app = typer.Typer()


def save_model_metadata(
    model_name: str,
    base_model: str,
    output_path: Path,
    description: str = "",
) -> None:
    """Salva os metadados do modelo ajustado em JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "base_model": base_model,
        "description": description,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.success(f"Metadados do modelo salvos em: {output_path}")


@app.command()
def main(
    train_dataset: Path = PROCESSED_DATA_DIR / "train_gemini.jsonl",
    val_dataset: Path | None = PROCESSED_DATA_DIR / "val_gemini.jsonl",
    base_model: str = DEFAULT_BASE_MODEL,
    display_name: str = "yan-assistant",
    epochs: int = 5,
    register_only: str | None = typer.Option(
        None,
        "--register-model",
        "-r",
        help="Se você treinou o modelo pela interface web do Google AI Studio, passe o nome do modelo aqui (ex: tunedModels/yan-assistant-xyz) para registrá-lo localmente.",
    ),
    output_info_path: Path = MODELS_DIR / "tuned_model_info.json",
):
    """Gerencia o Fine-Tuning do Gemini ou registra um modelo sintonizado no Google AI Studio."""
    if register_only:
        logger.info(f"Registrando modelo já sintonizado: {register_only}")
        save_model_metadata(
            model_name=register_only,
            base_model=base_model,
            output_path=output_info_path,
            description="Modelo sintonizado via Google AI Studio Web",
        )
        return

    api_key = get_gemini_api_key()
    if not train_dataset.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado em: {train_dataset}")

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        logger.error(
            "O pacote 'google-genai' não está instalado!\n"
            "Execute: uv add google-genai\n"
            "Ou treine pela interface web do Google AI Studio com o arquivo train_gemini.jsonl "
            "e registre com: python -m personal_assistant.modeling.train -r tunedModels/seu-modelo"
        )
        raise SystemExit(1) from e

    logger.info("Iniciando cliente da API do Gemini...")
    client = genai.Client(api_key=api_key)

    logger.info(f"Submetendo dataset de treino: {train_dataset}")
    examples = []
    with open(train_dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    logger.info(f"Total de exemplos carregados para o job: {len(examples)}")
    logger.info(f"Disparando Fine-Tuning do modelo base: {base_model}...")

    try:
        tuning_job = client.tunings.tune(
            base_model=base_model,
            training_dataset=types.TuningDataset(examples=examples),
            config=types.CreateTunedModelConfig(
                display_name=display_name,
                epoch_count=epochs,
            ),
        )
        logger.info(f"Job de Tuning submetido com sucesso! Nome: {tuning_job.name}")
        logger.info("Aguardando conclusão do treinamento (isso pode levar alguns minutos)...")

        while True:
            job_status = client.tunings.get(name=tuning_job.name)
            state = getattr(job_status, "state", "UNKNOWN")
            logger.info(f"Status atual do Tuning: {state}")

            if state in ("ACTIVE", "SUCCEEDED", "COMPLETED"):
                tuned_model_name = getattr(job_status, "tuned_model", tuning_job.name)
                logger.success(f"Treinamento concluído com sucesso! Modelo: {tuned_model_name}")
                save_model_metadata(
                    model_name=tuned_model_name,
                    base_model=base_model,
                    output_path=output_info_path,
                    description=f"Tuned via google-genai CLI com {epochs} épocas",
                )
                break
            elif state in ("FAILED", "CANCELLED"):
                logger.error(f"O treinamento falhou ou foi cancelado: {job_status}")
                break

            time.sleep(30)

    except Exception as err:  # noqa: BLE001
        logger.exception(f"Erro durante a chamada de tuning da API: {err}")
        logger.info(
            "Dica: Você também pode fazer o upload direto de 'data/processed/train_gemini.jsonl' "
            "no Google AI Studio (https://aistudio.google.com/) e depois registrar o modelo com:\n"
            "python -m personal_assistant.modeling.train -r tunedModels/nome-do-modelo"
        )


if __name__ == "__main__":
    app()
