"""
Módulo para processamento de chats exportados pelo WhatsApp.

Transforma conversas exportadas do WhatsApp em diálogos estruturados
compatíveis com formatos de fine-tuning (ex: OpenAI / ChatML).
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any

from loguru import logger
import typer

from personal_assistant.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(
    help="Ferramenta CLI para processar chat bruto do WhatsApp no formato de diálogo JSONL."
)

# Padrão para extrair timestamp, sender e conteúdo da mensagem
TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M|am|pm)?)\]\s*([^:]+):\s*(.*)$"
)
MEDIA_TAG_PATTERN = re.compile(r"<[^>]+>")
DELETED_MESSAGE_PATTERN = re.compile(r"mensagem\s+apagada", re.IGNORECASE)
URL_ONLY_PATTERN = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
PROMO_PATTERN = re.compile(
    r"(?:🔗\s*https?://|🏷\s*cupom|parcelado\s+\d+x|vendido e entregue pela amazon|#\w{8})",
    re.IGNORECASE,
)
FORWARDED_PREFIX_PATTERN = re.compile(r"\[Encaminhada\]\s*", re.IGNORECASE)
NOISE_CHARS = {".", "-", "..", "...", "?", "!", ",", "_", "~"}

# Categorias de palavras-chave para amostragem de diversidade
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "comida_restaurante": [
        "almoço",
        "almoçar",
        "comer",
        "comida",
        "chocolate",
        "cocada",
        "lanche",
        "fome",
        "restaurante",
        "pizza",
        "café",
    ],
    "faculdade_estudos": [
        "aula",
        "sala",
        "professor",
        "faculdade",
        "prova",
        "trabalho",
        "horário",
        "matéria",
        "senai",
        "p1",
        "estudar",
    ],
    "tecnologia_hardware": [
        "pc",
        "notebook",
        "placa",
        "processador",
        "rtx",
        "amd",
        "intel",
        "setup",
        "virtualização",
        "ram",
        "nvme",
        "ssd",
        "oracle",
        "windows",
        "linux",
    ],
    "financas_compras": [
        "nubank",
        "crédito",
        "cartão",
        "comprar",
        "dinheiro",
        "preço",
        "rendendo",
        "limite",
        "compra",
        "pix",
        "cupom",
        "banco",
        "reais",
    ],
    "jogos_games": [
        "jogo",
        "jogar",
        "gta",
        "cs",
        "steam",
        "game",
        "play",
        "ps5",
        "dualsense",
        "kratos",
        "troféu",
        "zerar",
    ],
    "transporte_viagem": [
        "carro",
        "viagem",
        "estrada",
        "rodovia",
        "km/h",
        "dirigir",
        "ssa",
        "uber",
        "vaga",
        "trânsito",
        "carona",
    ],
    "humor_cotidiano": [
        "kkk",
        "mano",
        "nego",
        "porra",
        "oxe",
        "carai",
        "foda",
        "vacilo",
        "pqp",
        "irmãozinho",
        "desgraça",
        "tá maluco",
    ],
}


def clean_message_text(text: str) -> str | None:
    """Cleans message text by removing media placeholders, system/deleted notices,

    promo spam, and meaningless noise. Returns None if the message should be discarded.
    """
    if not text:
        return None

    # Verifica por avisos de mensagem apagada
    if DELETED_MESSAGE_PATTERN.search(text):
        return None

    # Verifica por spam de cartão de promoção de afiliados
    if PROMO_PATTERN.search(text):
        return None

    # Remove placeholders de mídia (ex: <imagem ocultada>, <mensagem de voz omitida>)
    cleaned = MEDIA_TAG_PATTERN.sub("", text)

    # Remove tag [Encaminhada]
    cleaned = FORWARDED_PREFIX_PATTERN.sub("", cleaned)

    cleaned = cleaned.strip()

    # Descarta strings vazias, URLs nuas ou caracteres de ruído únicos
    if not cleaned:
        return None
    if URL_ONLY_PATTERN.match(cleaned):
        return None
    if cleaned in NOISE_CHARS:
        return None

    return cleaned


def parse_chat_file(file_path: Path) -> list[dict[str, Any]]:
    """Parses a WhatsApp chat export file into structured message dictionaries.

    Handles multiline messages, timestamps, and filters out system/encryption headers.
    """
    messages: list[dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = TIMESTAMP_PATTERN.match(line)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                sender = match.group(3).strip()
                content = match.group(4)

                dt_str = re.sub(r"[\u202f\xa0]", " ", f"{date_str} {time_str}")
                try:
                    dt = datetime.strptime(dt_str, "%m/%d/%y %I:%M:%S %p")  # noqa: DTZ007
                except ValueError:
                    try:
                        dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")  # noqa: DTZ007
                    except ValueError:
                        dt = datetime.now().astimezone()

                messages.append({"timestamp": dt, "sender": sender, "raw_content": content})
            else:
                # Continuação de mensagem multilinha
                if messages:
                    messages[-1]["raw_content"] += "\n" + line.rstrip("\r\n")

    # Limpa o conteúdo das mensagens
    cleaned_messages: list[dict[str, Any]] = []
    for msg in messages:
        cleaned_text = clean_message_text(msg["raw_content"])
        if cleaned_text is not None:
            cleaned_messages.append(
                {
                    "timestamp": msg["timestamp"],
                    "sender": msg["sender"],
                    "content": cleaned_text,
                }
            )

    return cleaned_messages


def segment_sessions(
    messages: list[dict[str, Any]], max_gap_minutes: float = 20.0
) -> list[list[dict[str, Any]]]:
    """Segments a chronological list of messages into conversational sessions

    based on a time difference threshold between consecutive messages.
    """
    if not messages:
        return []

    sessions: list[list[dict[str, Any]]] = []
    current_session: list[dict[str, Any]] = []
    gap_delta = timedelta(minutes=max_gap_minutes)

    for msg in messages:
        if not current_session:
            current_session.append(msg)
        else:
            time_gap = msg["timestamp"] - current_session[-1]["timestamp"]
            if time_gap < gap_delta:
                current_session.append(msg)
            else:
                sessions.append(current_session)
                current_session = [msg]

    if current_session:
        sessions.append(current_session)

    return sessions


def build_dialogues(
    sessions: list[list[dict[str, Any]]],
    assistant_sender: str = "Você",
    clone_name: str = "Yan Chagas",
    system_prompt: str | None = None,
    max_turns: int = 8,
) -> list[dict[str, Any]]:
    """
    Constrói diálogos padrão no formato JSONL a partir de sessões de conversa.

    - Mensagens consecutivas do usuário são mescladas em um único turno do usuário.
    - Mensagens consecutivas do assistente são mescladas em um único turno do assistente.
    - Remove turnos iniciais do assistente e turnos finais do usuário.
    - Mantém conversas com pelo menos 1 turno do usuário e 1 turno do assistente.
    - Opcionalmente limita o número de turnos para evitar drift de contexto.
    """
    if system_prompt is None:
        system_prompt = (
            f"Você é o clone digital de {clone_name}. "
            f"Responde direto ao ponto e de forma autêntica."
        )

    dialogues: list[dict[str, Any]] = []

    for session in sessions:
        senders = {m["sender"] for m in session}
        if assistant_sender not in senders or len(senders) < 2:
            continue

        collapsed_turns: list[dict[str, str]] = []
        for m in session:
            role = "assistant" if m["sender"] == assistant_sender else "user"
            content = m["content"]

            if not collapsed_turns or collapsed_turns[-1]["role"] != role:
                collapsed_turns.append({"role": role, "content": content})
            else:
                collapsed_turns[-1]["content"] += "\n" + content

        # Garante que a conversa comece com 'user'
        while collapsed_turns and collapsed_turns[0]["role"] != "user":
            collapsed_turns.pop(0)

        # Garante que a conversa termine com 'assistant'
        while collapsed_turns and collapsed_turns[-1]["role"] != "assistant":
            collapsed_turns.pop()

        if len(collapsed_turns) >= 2:
            # Limita o número máximo de turnos para manter as conversas focadas
            if len(collapsed_turns) > max_turns:
                collapsed_turns = collapsed_turns[:max_turns]
                if collapsed_turns[-1]["role"] != "assistant":
                    collapsed_turns.pop()

            dialogue = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *collapsed_turns,
                ]
            }
            dialogues.append(dialogue)

    return dialogues


def classify_dialogue_topics(dialogue: dict[str, Any]) -> list[str]:
    """Identifica tópicos presentes em um diálogo com base na ocorrência de palavras-chave."""
    text = " ".join(msg["content"].lower() for msg in dialogue["messages"][1:])
    matched = []
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in kws):
            matched.append(topic)
    return matched if matched else ["outros"]


def select_diverse_conversations(
    dialogues: list[dict[str, Any]],
    n: int = 25,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Seleciona um subconjunto balanceado e representativo de conversas entre diversos tópicos e comprimentos de conversação (turno único e múltiplos turnos).
    """
    if len(dialogues) <= n:
        return dialogues

    # Distribuição alvo para 25 conversas
    topic_targets: dict[str, int] = {
        "comida_restaurante": 3,
        "faculdade_estudos": 4,
        "tecnologia_hardware": 4,
        "financas_compras": 4,
        "jogos_games": 4,
        "transporte_viagem": 3,
        "humor_cotidiano": 3,
    }

    selected_indices: set[int] = set()
    selected_dialogues: list[dict[str, Any]] = []

    for topic, target_count in topic_targets.items():
        kws = TOPIC_KEYWORDS[topic]
        scored_candidates: list[tuple[float, int]] = []

        for idx, d in enumerate(dialogues):
            if idx in selected_indices:
                continue

            text = " ".join(msg["content"].lower() for msg in d["messages"][1:])
            kw_matches = sum(1 for kw in kws if kw in text)
            if kw_matches > 0:
                total_len = sum(len(m["content"]) for m in d["messages"][1:])
                has_http = "http" in text
                # Prefere diálogos conversacionais ricos sem conteúdo pesado de links
                score = (
                    kw_matches * 15.0 + min(total_len, 350) / 10.0 - (40.0 if has_http else 0.0)
                )
                scored_candidates.append((score, idx))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        picked = [idx for _, idx in scored_candidates[:target_count]]

        for p in picked:
            selected_indices.add(p)
            selected_dialogues.append(dialogues[p])

    # Fallback caso algum tópico fique sem representantes
    if len(selected_dialogues) < n:
        for idx, d in enumerate(dialogues):
            if idx not in selected_indices:
                selected_indices.add(idx)
                selected_dialogues.append(d)
                if len(selected_dialogues) == n:
                    break

    return selected_dialogues[:n]


def save_jsonl(dialogues: list[dict[str, Any]], output_path: Path) -> None:
    """Saves dialogues to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in dialogues)


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "chat.txt",
    output_path: Path = PROCESSED_DATA_DIR / "chat_dataset.jsonl",
    sample_output_path: Path = PROCESSED_DATA_DIR / "chat_dataset_sample25.jsonl",
    max_gap_minutes: float = 20.0,
    sample_size: int = 25,
    assistant_name: str = "Você",
    clone_name: str = "Yan Chagas",
) -> None:
    """Pipeline de CLI para transformar chat bruto do WhatsApp em dataset de diálogos em JSONL."""
    logger.info(f"Lendo e processando chat bruto de: {input_path}")
    if not input_path.exists():
        logger.error(f"Arquivo de entrada não encontrado: {input_path}")
        raise typer.Exit(code=1)

    cleaned_messages = parse_chat_file(input_path)
    logger.info(f"{len(cleaned_messages)} mensagens conversacionais limpas com sucesso.")

    sessions = segment_sessions(cleaned_messages, max_gap_minutes=max_gap_minutes)
    logger.info(
        f"Segmentado em {len(sessions)} sessões de conversa (intervalo máximo: {max_gap_minutes}min)."
    )

    dialogues = build_dialogues(
        sessions=sessions,
        assistant_sender=assistant_name,
        clone_name=clone_name,
    )
    logger.info(f"Construídos {len(dialogues)} diálogos válidos.")

    # Salva dataset completo
    save_jsonl(dialogues, output_path)
    logger.success(f"Dataset completo salvo em: {output_path} ({len(dialogues)} diálogos)")

    # Seleciona e salva amostra diversificada de 25 conversas
    sample_dialogues = select_diverse_conversations(dialogues, n=sample_size)
    save_jsonl(sample_dialogues, sample_output_path)
    logger.success(
        f"Amostra salva em: {sample_output_path} ({len(sample_dialogues)} diálogos diversificados)"
    )


if __name__ == "__main__":
    app()
