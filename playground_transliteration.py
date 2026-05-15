import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import gradio as gr
import torch
from transformers import EncoderDecoderModel, PreTrainedTokenizerFast

MODEL = "colesimmons/sumerian-transliteration"
ENC_TOK = "colesimmons/SumerianGlyphTokenizer_Roberta"
DEC_TOK = "colesimmons/SumerianTransliterationTokenizer_Roberta"
MAX_LENGTH = 128
NUM_BEAMS = 5

SAMPLE_GLYPHS = """ 
𒑏𒐈𒋡𒁉𒅆𒂟
𒑑𒐍𒋡𒁉𒁺
𒌓𒐋𒄰
𒑏𒐍𒈦𒋡𒁉𒅆𒂟
𒑒𒐌𒋡𒁉𒁺
𒌓𒐌𒄰
𒆠𒀭𒇋𒄰𒋫
𒁾𒉺𒋼𒋛𒅗
𒌗𒂡𒀭𒂄𒄀
𒈬𒍑𒊓𒀭𒋗𒀭𒂗𒍪𒈗𒋀𒀊𒆠𒈠𒆤𒂦𒈥𒌅𒈬𒆕
𒀭𒋗𒀭𒂗𒍪
𒈗𒆗𒂵
𒈗𒋀𒀊𒆠𒈠
𒈗𒀭𒌒𒁕𒇹𒁀
𒀀𒀀𒆗𒆷
𒉺𒋼𒋛
𒄑𒆵𒆠
"""


@dataclass(frozen=True)
class TransliterationComponents:
    model: EncoderDecoderModel
    encoder_tokenizer: PreTrainedTokenizerFast
    decoder_tokenizer: PreTrainedTokenizerFast
    device: torch.device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gradio dashboard for Sumerian transliteration.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA when it is available. By default the app runs on CPU.",
    )
    return parser.parse_args(argv)


def select_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")

        print("Warning: --use-gpu was passed, but CUDA is unavailable. Falling back to CPU.")

    return torch.device("cpu")


def load_fast_tokenizer(model_id: str) -> PreTrainedTokenizerFast:
    try:
        return PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception as online_error:
        try:
            return PreTrainedTokenizerFast.from_pretrained(
                model_id,
                local_files_only=True,
            )
        except Exception:
            raise online_error


def load_transliteration_components(use_gpu: bool) -> TransliterationComponents:
    device = select_device(use_gpu)

    model = EncoderDecoderModel.from_pretrained(MODEL)
    encoder_tokenizer = load_fast_tokenizer(ENC_TOK)
    decoder_tokenizer = load_fast_tokenizer(DEC_TOK)

    model.eval()
    model.to(device)

    return TransliterationComponents(
        model=model,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        device=device,
    )


def transliterate(glyphs: str, components: TransliterationComponents) -> str:
    if not glyphs.strip():
        return ""

    encoded = components.encoder_tokenizer(
        glyphs,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(components.device)

    with torch.no_grad():
        output_ids = components.model.generate(
            **encoded,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            decoder_start_token_id=components.model.config.decoder_start_token_id,
        )

    text = components.decoder_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text.replace("▁", " ").strip()


def create_app(components: TransliterationComponents) -> gr.Interface:
    def predict(glyphs: str) -> str:
        return transliterate(glyphs, components)

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            label="шумерские знаки",
            value=SAMPLE_GLYPHS,
            lines=18,
        ),
        outputs=gr.Textbox(
            label="результат работы модели",
            lines=8,
        ),
        title="Sumerian Transliteration",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    components = load_transliteration_components(use_gpu=args.use_gpu)
    demo = create_app(components)
    demo.launch()


if __name__ == "__main__":
    main()