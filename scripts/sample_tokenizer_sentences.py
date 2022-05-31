import os
import argparse
import random
from tqdm import tqdm
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample Sentences from monolingual corpora to train tokenizer"
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        required=True,
        help="Path containing monolingual corpora for different languages",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to store sampled sentences"
    )
    parser.add_argument("--alpha", type=float, default=0.3, help="multinomial alpha")
    parser.add_argument("--seed", type=int, default=10, help="random seed")

    return parser


def calc_num_samples_sentences(
    lang_num_lines: Dict[str, int], alpha: float
) -> Dict[str, int]:
    lang_prob = {}

    total_sentences = sum(lang_num_lines.values())
    for key, value in lang_num_lines.items():
        lang_prob[key] = value / total_sentences

    total_distr = 0
    for k, v in lang_prob.items():
        total_distr += v**alpha

    new_prob = {k: v**alpha / total_distr for k, v in lang_prob.items()}

    sampled_sentences = {}

    for language, num_lines in lang_num_lines.items():
        for lang_code, sampled_prob in new_prob.items():
            if language == lang_code:
                num_sentences = sampled_prob * num_lines
                sampled_sentences[language] = round(num_sentences)

    return sampled_sentences


def main():

    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    logger.info("***** Sampling Sentences for Tokenizer Training *****")

    files = [
        os.path.join(args.datasets_path, file)
        for file in os.listdir(args.datasets_path)
    ]

    logger.info(f"Number of training files found: {len(files)}")

    lang_corpus = {}
    lang_num_lines = {}

    for file in files:
        lang_code = file.split(".")[-1]
        with open(file) as f:
            txt = f.readlines()
            lang_corpus[lang_code] = txt
            lang_num_lines[lang_code] = len(txt)

    sampled_sentences = calc_num_samples_sentences(lang_num_lines, args.alpha)

    for lang in tqdm(sampled_sentences.keys()):
        logger.info(
            f"Number of sampled sentences for {lang} = {sampled_sentences[lang]}"
        )
        sentences = random.sample(lang_corpus[lang], sampled_sentences[lang])
        file = os.path.join(args.output_path, "sampled." + lang)

        with open(file, "w") as out_file:
            out_file.writelines(sentences)


if __name__ == "__main__":
    main()
