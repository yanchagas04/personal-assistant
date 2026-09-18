from pathlib import Path
import re

from loguru import logger
import pandas as pd
import typer

from personal_assistant.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def parse_whatsapp_chat(file_path: str | Path) -> pd.DataFrame:
    """Lê um arquivo de exportação de chat do WhatsApp (.txt) e retorna

    um DataFrame com as colunas: datetime, sender, message, etc.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    pattern = re.compile(
        r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}:\d{2}(?:\s*[APap][Mm])?)\]\s*(?:(?P<sender>[^:]+?):\s+)?(?P<message>.*)$"
    )

    records = []
    current_msg = None

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        for line in f:
            line_clean = line.rstrip("\r\n")
            match = pattern.match(line_clean)

            if match:
                if current_msg is not None:
                    records.append(current_msg)

                data = match.groupdict()
                sender = data["sender"]
                msg_text = data["message"]

                if sender is None:
                    sender = "Sistema"

                current_msg = {
                    "date": data["date"],
                    "time": data["time"],
                    "sender": sender.strip(),
                    "message": msg_text,
                }
            else:
                if current_msg is not None:
                    current_msg["message"] += "\n" + line_clean

    if current_msg is not None:
        records.append(current_msg)

    df = pd.DataFrame(records)

    if not df.empty:
        datetime_str = (df["date"] + " " + df["time"]).str.replace(r"\s+", " ", regex=True)
        try:
            df["datetime"] = pd.to_datetime(
                datetime_str, format="%m/%d/%y %I:%M:%S %p", errors="coerce"
            )
        except (ValueError, TypeError):
            df["datetime"] = pd.to_datetime(datetime_str, format="mixed", errors="coerce")

        columns_order = ["datetime", "sender", "message", "date", "time"]
        df = df[[c for c in columns_order if c in df.columns]]

    return df


def clean_chat_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra mensagens de mídia, encaminhadas, vazias e sem conteúdo alfanumérico."""
    if df.empty:
        return df

    media_pattern = (
        r"<imagem ocultada>|<mensagem de voz omitida>|<figurinha omitida>|"
        r"<vídeo omitido>|<áudio ocultado>|<arquivo anexado>"
    )
    is_media = df["message"].str.contains(media_pattern, regex=True)
    is_forwarded = df["message"].str.contains(r"Encaminhada", regex=True)

    df_clean = df[~is_media & ~is_forwarded].copy()

    cols = [c for c in ["datetime", "sender", "message"] if c in df_clean.columns]
    df_clean = df_clean[cols].dropna(subset=["datetime", "message"])

    # Remove mensagens que não contenham pelo menos um caractere alfanumérico (ex: '.', '...', '!')
    has_alphanumeric = df_clean["message"].str.contains(r"\w", regex=True)
    df_clean = df_clean[has_alphanumeric]

    return df_clean.reset_index(drop=True)


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "chat.txt",
    output_path: Path = PROCESSED_DATA_DIR / "chat_processed.csv",
):
    """Carrega o chat bruto do WhatsApp, aplica a limpeza e salva em data/processed/."""
    logger.info(f"Lendo dados brutos de: {input_path}")
    df_raw = parse_whatsapp_chat(input_path)
    logger.info(f"Total de mensagens lidas: {len(df_raw)}")

    df_clean = clean_chat_dataframe(df_raw)
    logger.info(f"Total após limpeza: {len(df_clean)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    logger.success(f"Dados salvos com sucesso em: {output_path}")


if __name__ == "__main__":
    app()
