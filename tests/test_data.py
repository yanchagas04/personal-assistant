import json
from pathlib import Path
import pandas as pd
import pytest

from personal_assistant.dataset import clean_chat_dataframe
from personal_assistant.features import (
    build_gemini_conversations,
    export_jsonl,
    merge_consecutive_messages,
)


def test_clean_chat_dataframe():
    data = {
        "datetime": pd.to_datetime([
            "2026-08-19 10:00:00",
            "2026-08-19 10:01:00",
            "2026-08-19 10:02:00",
            "2026-08-19 10:03:00",
            "2026-08-19 10:04:00",
        ]),
        "sender": ["Danilo", "Você", "Vinio", "Você", "Você"],
        "message": [
            "Bora almoçar?",
            "<imagem ocultada>",
            "Encaminhada: Olha isso",
            ".",
            "Bora sim!",
        ],
    }
    df = pd.DataFrame(data)
    df_clean = clean_chat_dataframe(df)

    # Apenas 'Bora almoçar?' e 'Bora sim!' devem restar
    assert len(df_clean) == 2
    assert list(df_clean["message"]) == ["Bora almoçar?", "Bora sim!"]


def test_merge_consecutive_messages():
    data = {
        "datetime": pd.to_datetime([
            "2026-08-19 10:00:00",
            "2026-08-19 10:00:30",
            "2026-08-19 10:05:00",
        ]),
        "sender": ["Você", "Você", "Danilo"],
        "message": ["Primeira parte", "Segunda parte", "Beleza"],
    }
    df = pd.DataFrame(data)
    df_merged = merge_consecutive_messages(df, max_gap_seconds=120)

    assert len(df_merged) == 2
    assert df_merged.iloc[0]["message"] == "Primeira parte\nSegunda parte"
    assert df_merged.iloc[1]["message"] == "Beleza"


def test_build_gemini_conversations():
    data = {
        "datetime": pd.to_datetime([
            "2026-08-19 10:00:00",
            "2026-08-19 10:01:00",
            "2026-08-19 10:02:00",
            # Quebra de sessão (intervalo > 20 min)
            "2026-08-19 11:00:00",
            "2026-08-19 11:01:00",
        ]),
        "sender": ["Danilo", "Vitu", "Você", "Danilo", "Você"],
        "message": [
            "Chegou o lanche",
            "Bora comer",
            "Tô descendo!",
            "E o código?",
            "Tá pronto já",
        ],
    }
    df = pd.DataFrame(data)
    examples = build_gemini_conversations(
        df,
        target_sender="Você",
        session_threshold_minutes=20,
        max_context_messages=2,
    )

    assert len(examples) == 2

    # Primeiro exemplo
    ex1 = examples[0]
    assert ex1["messages"][0]["role"] == "user"
    assert "Danilo: Chegou o lanche" in ex1["messages"][0]["content"]
    assert "Vitu: Bora comer" in ex1["messages"][0]["content"]
    assert ex1["messages"][1]["role"] == "model"
    assert ex1["messages"][1]["content"] == "Tô descendo!"

    # Segundo exemplo (após a quebra de sessão)
    ex2 = examples[1]
    assert ex2["messages"][0]["content"] == "Danilo: E o código?"
    assert ex2["messages"][1]["content"] == "Tá pronto já"


def test_export_jsonl(tmp_path: Path):
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Danilo: Olá"},
                {"role": "model", "content": "Opa!"},
            ]
        }
    ]
    file_path = tmp_path / "train.jsonl"
    export_jsonl(examples, file_path)

    assert file_path.exists()
    lines = file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    loaded = json.loads(lines[0])
    assert "messages" in loaded
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][0]["role"] == "user"
    assert loaded["messages"][1]["role"] == "model"
