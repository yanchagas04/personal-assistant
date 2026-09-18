import json
from pathlib import Path
import re
from typing import Any

from loguru import logger
import pandas as pd
import typer

from personal_assistant.config import PROCESSED_DATA_DIR

app = typer.Typer()


def merge_consecutive_messages(df: pd.DataFrame, max_gap_seconds: int = 120) -> pd.DataFrame:
    """Agrupa mensagens consecutivas enviadas pelo mesmo autor em um intervalo curto de tempo."""
    if df.empty:
        return df

    df = df.sort_values("datetime").reset_index(drop=True)
    merged_records = []

    current_row = None

    for _, row in df.iterrows():
        if current_row is None:
            current_row = row.to_dict()
            continue

        same_sender = row["sender"] == current_row["sender"]
        time_diff = (row["datetime"] - current_row["datetime"]).total_seconds()

        if same_sender and time_diff <= max_gap_seconds:
            current_row["message"] = f"{current_row['message']}\n{row['message']}"
        else:
            merged_records.append(current_row)
            current_row = row.to_dict()

    if current_row is not None:
        merged_records.append(current_row)

    return pd.DataFrame(merged_records)


def is_valid_target_message(msg: str) -> bool:
    """Valida se a mensagem possui conteúdo alfanumérico válido para treino."""
    msg_clean = msg.strip()
    if not msg_clean:
        return False
    return bool(re.search(r"\w", msg_clean))


def build_gemini_conversations(
    df: pd.DataFrame,
    target_sender: str = "Você",
    session_threshold_minutes: int = 20,
    max_context_messages: int = 4,
) -> list[dict[str, Any]]:
    """Gera exemplos de conversas formatados para Fine-Tuning do Gemini.

    Formato de cada exemplo:
    {
      "messages": [
        {"role": "user", "content": "Danilo: mensagem\nVitu: mensagem"},
        {"role": "model", "content": "Sua resposta"}
      ]
    }
    """
    if df.empty:
        return []

    df_merged = merge_consecutive_messages(df)

    examples: list[dict[str, Any]] = []
    current_session_msgs: list[dict[str, Any]] = []
    last_datetime = None

    for _, row in df_merged.iterrows():
        dt = row["datetime"]
        sender = row["sender"]
        message = str(row["message"]).strip()

        # Verifica se estourou o limite da sessão
        if last_datetime is not None:
            gap_minutes = (dt - last_datetime).total_seconds() / 60.0
            if gap_minutes > session_threshold_minutes:
                current_session_msgs = []

        last_datetime = dt

        if sender == target_sender:
            # O autor alvo respondeu!
            # Só cria o exemplo se houver contexto anterior nesta sessão
            if current_session_msgs and is_valid_target_message(message):
                # Filtra mensagens do histórico recente
                context_slice = current_session_msgs[-max_context_messages:]
                context_lines = [
                    f"{msg['sender']}: {msg['message']}"
                    for msg in context_slice
                    if is_valid_target_message(msg["message"])
                ]

                if context_lines:
                    context_str = "\n".join(context_lines)
                    examples.append(
                        {
                            "messages": [
                                {"role": "user", "content": context_str},
                                {"role": "model", "content": message},
                            ]
                        }
                    )

            # Mantém no histórico da sessão
            current_session_msgs.append({"sender": sender, "message": message})
        else:
            # Mensagem de outro participante
            current_session_msgs.append({"sender": sender, "message": message})

    return examples


def export_jsonl(examples: list[dict[str, Any]], filepath: Path) -> None:
    """Salva uma lista de exemplos no formato JSON Lines (JSONL)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "chat_processed.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    target_sender: str = "Você",
    train_ratio: float = 0.85,
    session_threshold_minutes: int = 20,
    max_context_messages: int = 4,
):
    """Lê o CSV processado e gera os arquivos train_gemini.jsonl e val_gemini.jsonl."""
    logger.info(f"Lendo base processada de: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    df["datetime"] = pd.to_datetime(df["datetime"])

    logger.info(f"Total de registros carregados: {len(df)}")
    examples = build_gemini_conversations(
        df,
        target_sender=target_sender,
        session_threshold_minutes=session_threshold_minutes,
        max_context_messages=max_context_messages,
    )
    logger.info(f"Total de pares de conversação gerados: {len(examples)}")

    if not examples:
        logger.warning("Nenhum exemplo foi gerado. Verifique o target_sender e os dados.")
        return

    # Divisão treino e validação
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    train_path = output_dir / "train_gemini.jsonl"
    val_path = output_dir / "val_gemini.jsonl"

    export_jsonl(train_examples, train_path)
    export_jsonl(val_examples, val_path)

    logger.success(
        f"Exportado com sucesso!\n"
        f" - Treino: {len(train_examples)} exemplos salvos em {train_path}\n"
        f" - Validação: {len(val_examples)} exemplos salvos em {val_path}"
    )


if __name__ == "__main__":
    app()
