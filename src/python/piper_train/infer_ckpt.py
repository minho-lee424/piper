import argparse
import json
import logging
import os
from enum import Enum
from typing import List

import numpy as np
import pyopenjtalk
import torch
from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run

from piper_train.vits.lightning import VitsModel
from piper_train.vits.utils import audio_float_to_int16
from piper_train.vits.wavfile import write as write_wav

# Logger setup
_LOGGER = logging.getLogger("piper_train.infer")

PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence


class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    TEXT = "text"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-s", "--sid", default=0, type=int)
    parser.add_argument("-l", "--lang")
    parser.add_argument("-t", "--text", required=True)
    parser.add_argument("-o", "--output_path", default="infer/test.wav")
    args = parser.parse_args()
    return args


def load_model_and_config(model_path, config_path):
    """Load the model and its configuration."""
    _LOGGER.debug(f"Loading model from {model_path}")
    with open(config_path, "r") as file:
        config = json.load(file)

    model = VitsModel.load_from_checkpoint(model_path, dataset=None)
    model.eval()

    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

    _LOGGER.info(f"Model loaded from {model_path}")
    return model, config


def inferencing(
    model,
    config,
    sid,
    lang,
    line,
    output_path,
    length_scale=1,
    noise_scale=0.667,
    noise_scale_w=0.8,
):
    audios = []
    text = phonemize(config, lang, line)

    for phonemes in text:
        phoneme_ids = phonemes_to_ids(config, phonemes)
        num_speakers = config["num_speakers"]
        speaker_id = sid if num_speakers > 1 else None
        text = torch.LongTensor(phoneme_ids).unsqueeze(0)
        text_lengths = torch.LongTensor([len(phoneme_ids)])
        scales = [noise_scale, length_scale, noise_scale_w]
        sid_tensor = torch.LongTensor([speaker_id]) if speaker_id is not None else None
        audio = model(text, text_lengths, scales, sid=sid_tensor).detach().numpy()
        audio = audio_float_to_int16(audio.squeeze())
        audios.append(audio)

    merged_audio = np.concatenate(audios)
    sample_rate = config["audio"]["sample_rate"]
    write_wav(output_path, sample_rate, merged_audio)


def phonemize(config, lang, text: str) -> List[List[str]]:
    """Text to phonemes grouped by sentence."""
    if config["phoneme_type"] == PhonemeType.ESPEAK:
        lang = lang or config["espeak"]["voice"]
        if lang == "ar":
            text = tashkeel_run(text)
        if lang == "ja":
            text = pyopenjtalk.g2p(text, kana=True)
        return phonemize_espeak(text, lang)
    if config["phoneme_type"] == PhonemeType.TEXT:
        return phonemize_codepoints(text)
    raise ValueError(f"Unexpected phoneme type: {config.phoneme_type}")


def phonemes_to_ids(config, phonemes: List[str]) -> List[int]:
    """Phonemes to ids."""
    id_map = config["phoneme_id_map"]
    ids: List[int] = list(id_map[BOS])
    for phoneme in phonemes:
        if phoneme not in id_map:
            print(f"Missing phoneme from id map: {phoneme}")
            continue
        ids.extend(id_map[phoneme])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    args = get_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    model, config = load_model_and_config(args.model, args.config)
    inferencing(model, config, args.sid, args.lang, args.text, args.output_path)


if __name__ == "__main__":
    main()
