import subprocess
from pathlib import Path

import sentencepiece as spm

prefix = Path('../dataset/05-translation/ted2020').absolute()
vocab_size = 8000


def train_sentencepiece_model(src_lang, tgt_lang):
    if (prefix / f'spm{vocab_size}.model').exists():
        print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
    else:
        spm.SentencePieceTrainer.train(
            input=','.join([f'{prefix}/train.clean.{src_lang}',
                            f'{prefix}/valid.clean.{src_lang}',
                            f'{prefix}/train.clean.{tgt_lang}',
                            f'{prefix}/valid.clean.{tgt_lang}']),
            model_prefix=prefix / f'spm{vocab_size}',
            vocab_size=vocab_size,
            character_coverage=1,
            model_type='unigram',  # 'bpe' works as well
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )


def embedding(src_lang, tgt_lang):
    spm_model = spm.SentencePieceProcessor(model_file=str(prefix / f'spm{vocab_size}.model'))
    in_tag = {
        'train': 'train.clean',
        'valid': 'valid.clean'
    }
    for split in ['train', 'valid']:
        for lang in [src_lang, tgt_lang]:
            out_path = prefix / f'{split}.{lang}'
            if out_path.exists():
                print(f"{out_path} exists. skipping spm_encode.")
            else:
                with open(prefix / f'{split}.{lang}', 'w', encoding='utf-8') as out_f:
                    with open(prefix / f'{in_tag[split]}.{lang}', 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            line = line.strip()
                            tok = spm_model.encode(line, out_type=str)
                            print(' '.join(tok), file=out_f)


def fairseq_data():
    binpath = Path('../dataset/05-translation/data-bin', 'ted2020')

    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
    else:
        command = [
            'python', '-m', 'fairseq_cli.preprocess',
            '--source-lang', src_lang,
            '--target-lang', tgt_lang,
            '--trainpref', str(prefix / 'train'),
            '--validpref', str(prefix / 'valid'),
            '--destdir', str(binpath),
            '--joined-dictionary',
            '--workers', '2'
        ]

        subprocess.run(command)


if __name__ == '__main__':
    src_lang, tgt_lang = 'en', 'zh'
    train_sentencepiece_model(src_lang, tgt_lang)
    embedding(src_lang, tgt_lang)
    fairseq_data()
