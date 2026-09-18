import json
from pathlib import Path

from loguru import logger
import typer

from personal_assistant.config import MODELS_DIR, get_gemini_api_key

app = typer.Typer()


def load_model_name(info_path: Path = MODELS_DIR / "tuned_model_info.json") -> str:
    """Lê o nome do modelo sintonizado a partir do arquivo de metadados."""
    if not info_path.exists():
        raise FileNotFoundError(
            f"Arquivo de metadados não encontrado em: {info_path}.\n"
            "Execute o treinamento primeiro ou registre o modelo com:\n"
            "python -m personal_assistant.modeling.train -r tunedModels/seu-modelo"
        )
    with open(info_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["model_name"]


def generate_reply(client, model_name: str, prompt: str) -> str:
    """Gera uma resposta utilizando o modelo do Gemini."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text.strip() if response.text else ""


@app.command()
def main(
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Nome do modelo sintonizado (ex: tunedModels/yan-assistant-xyz). Se omitido, lê de models/tuned_model_info.json.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Mensagem para testar uma única resposta. Se omitido, inicia o modo chat interativo.",
    ),
    info_path: Path = MODELS_DIR / "tuned_model_info.json",
):
    """Executa a inferência com o modelo sintonizado do Yan."""
    api_key = get_gemini_api_key()

    try:
        from google import genai
    except ImportError as e:
        logger.error("O pacote 'google-genai' não está instalado!\nExecute: uv add google-genai")
        raise SystemExit(1) from e

    target_model = model_name or load_model_name(info_path)
    logger.info(f"Conectando ao modelo: {target_model}")

    client = genai.Client(api_key=api_key)

    if prompt:
        logger.info(f"Pergunta: {prompt}")
        reply = generate_reply(client, target_model, prompt)
        print(f"\n[Yan Bot]: {reply}\n")
        return

    print("\n" + "=" * 50)
    print("🤖 Modo Chat Interativo com o Yan Bot")
    print("Digite sua mensagem para simular uma conversa com o Yan.")
    print("Digite 'sair' ou pressione Ctrl+C para encerrar.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Você: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                print("Até mais!")
                break

            reply = generate_reply(client, target_model, user_input)
            print(f"\nYan: {reply}\n")
        except KeyboardInterrupt:
            print("\nEncerrando chat...")
            break
        except Exception as err:  # noqa: BLE001
            logger.error(f"Erro ao gerar resposta: {err}")


if __name__ == "__main__":
    app()
