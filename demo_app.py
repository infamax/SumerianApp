import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
)

MODEL = "colesimmons/sumerian-transliteration"
ENC_TOK = "colesimmons/SumerianGlyphTokenizer_Roberta"
DEC_TOK = "colesimmons/SumerianTransliterationTokenizer_Roberta"
MAX_LENGTH = 128
NUM_BEAMS = 5
CPU_NUM_BEAMS = 1
TRANSLITERATION_MAX_NEW_TOKENS = 64
TRANSLATION_SOURCE_PREFIX = "Переведи с шумерского на русский: "
TRANSLATION_MAX_SOURCE_LENGTH = 128
TRANSLATION_MAX_NEW_TOKENS = 128
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_OCR_MODEL_PATH = "ocr_model_weights"
DEFAULT_TRANSLATION_MODEL_PATH = "translator_model_weights"
TRANSLITERATION_MODE = "Модель транслитерации"
TRANSLATION_MODE = "Модель перевода"
TRANSLITERATION_TRANSLATION_MODE = "Транслитерация + перевод"
OCR_MODE = "OCR"
SAMPLE_TRANSLITERATION = "a-a den-ki za3-mi2-zu dug3-ga"

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


@dataclass(frozen=True)
class OcrComponents:
    model: Any
    processor: Any
    device: torch.device


@dataclass(frozen=True)
class TranslationComponents:
    model: Any
    tokenizer: Any
    device: torch.device


@dataclass(frozen=True)
class AppComponents:
    transliteration: TransliterationComponents
    ocr_model_path: str
    translation_model_path: str
    use_gpu: bool
    ocr: OcrComponents | None = None
    translation: TranslationComponents | None = None

    def get_ocr(self) -> OcrComponents:
        if self.ocr is None:
            object.__setattr__(
                self,
                "ocr",
                load_ocr_components(
                    model_path=self.ocr_model_path,
                    use_gpu=self.use_gpu,
                ),
            )

        return self.ocr

    def get_translation(self) -> TranslationComponents:
        if self.translation is None:
            object.__setattr__(
                self,
                "translation",
                load_translation_components(
                    model_path=self.translation_model_path,
                    use_gpu=self.use_gpu,
                ),
            )

        return self.translation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gradio dashboard for Sumerian transliteration.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA when it is available. By default the app runs on CPU.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind the Gradio server to. Defaults to {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port to bind the Gradio server to. Defaults to {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--ocr-model-path",
        default=DEFAULT_OCR_MODEL_PATH,
        help="Path or Hugging Face model ID for the OCR model.",
    )
    parser.add_argument(
        "--translation-model-path",
        default=DEFAULT_TRANSLATION_MODEL_PATH,
        help="Path or Hugging Face model ID for the translation model weights.",
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
        return PreTrainedTokenizerFast.from_pretrained(
            model_id,
            local_files_only=True,
        )
    except Exception:
        return PreTrainedTokenizerFast.from_pretrained(model_id)


def load_encoder_decoder_model(model_id: str) -> EncoderDecoderModel:
    try:
        return EncoderDecoderModel.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return EncoderDecoderModel.from_pretrained(model_id)


def load_transliteration_components(use_gpu: bool) -> TransliterationComponents:
    device = select_device(use_gpu)

    model = load_encoder_decoder_model(MODEL)
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


def load_ocr_components(model_path: str, use_gpu: bool) -> OcrComponents:
    device = select_device(use_gpu)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    model.eval()

    return OcrComponents(
        model=model,
        processor=processor,
        device=device,
    )


def load_translation_components(model_path: str, use_gpu: bool) -> TranslationComponents:
    device = select_device(use_gpu)
    model_location = Path(model_path) if Path(model_path).exists() else model_path

    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_location)
    model.to(device)
    model.eval()

    return TranslationComponents(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def load_app_components(
    ocr_model_path: str,
    translation_model_path: str,
    use_gpu: bool,
) -> AppComponents:
    return AppComponents(
        transliteration=load_transliteration_components(use_gpu=use_gpu),
        ocr_model_path=ocr_model_path,
        translation_model_path=translation_model_path,
        use_gpu=use_gpu,
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

    generation_kwargs = {
        "max_new_tokens": TRANSLITERATION_MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS if components.device.type == "cuda" else CPU_NUM_BEAMS,
        "decoder_start_token_id": components.model.config.decoder_start_token_id,
        "eos_token_id": components.model.config.eos_token_id,
        "pad_token_id": components.model.config.pad_token_id,
    }
    if generation_kwargs["num_beams"] > 1:
        generation_kwargs["early_stopping"] = True

    with torch.no_grad():
        output_ids = components.model.generate(
            **encoded,
            **generation_kwargs,
        )

    text = components.decoder_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text.replace("▁", " ").strip()


def translate_transliteration(
    transliteration: str,
    components: TranslationComponents,
) -> str:
    if not transliteration.strip():
        return ""

    encoded = components.tokenizer(
        [TRANSLATION_SOURCE_PREFIX + transliteration.strip()],
        max_length=TRANSLATION_MAX_SOURCE_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(components.device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": TRANSLATION_MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS if components.device.type == "cuda" else CPU_NUM_BEAMS,
    }
    if generation_kwargs["num_beams"] > 1:
        generation_kwargs["early_stopping"] = True

    with torch.no_grad():
        generated = components.model.generate(**encoded, **generation_kwargs)

    return components.tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()


def run_ocr(image: Image.Image, components: OcrComponents) -> str:
    rgb_image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image},
                {"type": "text", "text": "OCR:"},
            ],
        },
    ]

    inputs = components.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(components.device)

    with torch.no_grad():
        output_ids = components.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=32,
            repetition_penalty=1.03,
        )

    predicted_ids = output_ids[0][inputs["input_ids"].shape[1] :][:-1].tolist()
    return components.processor.decode(
        predicted_ids,
        skip_special_tokens=False,
    ).strip()


def create_app(components: AppComponents) -> gr.Blocks:
    def update_mode(selected_model: str) -> tuple[gr.update, gr.update]:
        text_input_visible = selected_model != OCR_MODE
        text_input_label = (
            "транслитерация"
            if selected_model == TRANSLATION_MODE
            else "шумерские знаки"
        )
        text_input_value = (
            SAMPLE_TRANSLITERATION
            if selected_model == TRANSLATION_MODE
            else SAMPLE_GLYPHS
        )
        return (
            gr.update(
                visible=text_input_visible,
                label=text_input_label,
                value=text_input_value,
            ),
            gr.update(visible=selected_model == OCR_MODE),
        )

    def predict(selected_model: str, glyphs: str, image: Image.Image | None) -> str:
        if selected_model == TRANSLITERATION_MODE:
            if not glyphs.strip():
                return "Введите шумерские знаки."

            return transliterate(glyphs, components.transliteration)

        if selected_model == TRANSLATION_MODE:
            if not glyphs.strip():
                return "Введите транслитерацию."

            return translate_transliteration(glyphs, components.get_translation())

        if selected_model == TRANSLITERATION_TRANSLATION_MODE:
            if not glyphs.strip():
                return "Введите шумерские знаки."

            transliteration = transliterate(glyphs, components.transliteration)
            translation = translate_transliteration(
                transliteration,
                components.get_translation(),
            )
            return (
                f"Транслитерация:\n{transliteration}\n\n"
                f"Перевод:\n{translation}"
            )

        if selected_model == OCR_MODE:
            if image is None:
                return "Загрузите изображение для OCR."

            return run_ocr(image, components.get_ocr())

        return "Выберите модель."

    with gr.Blocks(title="Sumerian Models") as demo:
        gr.Markdown("# Sumerian Models")
        model_selector = gr.Radio(
            choices=[
                TRANSLITERATION_MODE,
                TRANSLATION_MODE,
                TRANSLITERATION_TRANSLATION_MODE,
                OCR_MODE,
            ],
            value=TRANSLITERATION_MODE,
            label="Выбор модели",
        )
        glyph_input = gr.Textbox(
            label="шумерские знаки",
            value=SAMPLE_GLYPHS,
            lines=18,
            visible=True,
        )
        image_input = gr.Image(
            label="изображение",
            type="pil",
            visible=False,
        )
        output = gr.Textbox(
            label="результат работы модели",
            lines=8,
        )
        run_button = gr.Button("Запустить модель")

        model_selector.change(
            fn=update_mode,
            inputs=model_selector,
            outputs=[glyph_input, image_input],
        )
        run_button.click(
            fn=predict,
            inputs=[model_selector, glyph_input, image_input],
            outputs=output,
        )

    return demo


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    components = load_app_components(
        ocr_model_path=args.ocr_model_path,
        translation_model_path=args.translation_model_path,
        use_gpu=args.use_gpu,
    )
    demo = create_app(components)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()