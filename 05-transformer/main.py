from pathlib import Path

from corpus import clean_corpus, split_train_valid
from utils.settings import same_seeds

if __name__ == '__main__':
    same_seeds(73)

    src_lang = 'en'
    tgt_lang = 'zh'

    data_dir = '../dataset/05-translation/ted2020'
    prefix = Path(data_dir).absolute()
    data_prefix = f'{prefix}/train_dev.raw'

    clean_corpus(data_prefix, src_lang, tgt_lang)
    # clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)
    split_train_valid(data_dir, data_prefix, prefix, src_lang, tgt_lang)
