import json
import random

import torch
from torch.utils.data import Dataset, DataLoader


def read_data(file):
    with open(file, 'r', encoding='UTF-8') as reader:
        data = json.load(reader)
    return data['questions'], data['paragraphs']


def questions_paragraphs_tokenized(tokenizer, file):
    questions, paragraphs = read_data(file)
    # 对训练问题进行分词处理，add_special_tokens=False表示不添加特殊标记
    # 得到BatchEncoding对象
    # data->input_ids:输入序列的token的ids
    # data->token_type_ids:用于区分输入序列中不同部分的标记类型ID，比如问答中，输入序列被分为2部分，0是question,1是paragraph
    # data->attention_mask:用于掩码输入序列中填充部分的注意力掩码
    # Encoding(num_tokens=10, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
    # encodings->num_tokens:多少个token  ids对应input_ids  type_ids对应token_type_ids tokens对应编码前的数据 offsets对应每个token的开始与结束 words对应下表索引
    questions_tokenized = tokenizer([question['question_text'] for question in questions], add_special_tokens=False)
    paragraphs_tokenized = tokenizer(paragraphs, add_special_tokens=False)
    return questions, questions_tokenized, paragraphs_tokenized


def get_data_loader(split='train', batch_size=16, questions=None, questions_tokenized=None, paragraphs_tokenized=None, tokenizer=None, shuffle=True):
    """

    :param split:
    :param batch_size:
        Note: Do NOT change batch size of dev_loader / test_loader !
        Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    :param questions:
    :param questions_tokenized:
    :param paragraphs_tokenized:
    :return:
    """
    data_set = QADataset(split, questions, questions_tokenized, paragraphs_tokenized, tokenizer)
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


class QADataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs, tokenizer):
        self.tokenizer = tokenizer
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 150
        # 文本幅度？？？
        self.doc_stride = 75
        # 输入seq的长度=[CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        tokenized_question = self.tokenized_questions[index]
        tokenized_paragraph = self.tokenized_paragraphs[question['paragraph_id']]
        # result = '' if question['answer_start'] is None else self.tokenizer.decode(tokenized_paragraph.ids[question['answer_start']: question['answer_end'] + 1])
        # print(f"index:{index}, question:{question}, answer:{result}")
        # 如何做预处理，如何防止模型学习到不应该学习到的东西
        if self.split == 'train':
            # 使用字符的位置，找到标记的问题，如果是每个字符就是一个token，则数据是相等的
            answer_start_token = tokenized_paragraph.char_to_token(question['answer_start'])
            answer_end_token = tokenized_paragraph.char_to_token(question['answer_end'])
            # mid = (answer_start_token + answer_end_token) // 2
            # paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            # 答案结尾的索引answer_end_token - 片段最大长度self.max_paragraph_len
            # 如果小于0，paragraph_start随机范围以0开始，如果大于0，以大于0的值开始
            # paragraph_start的随机范围以answer_start_token结束，最大为len(tokenized_paragraph) - self.max_paragraph_len
            paragraph_start = random.randint(max(0, answer_end_token - self.max_paragraph_len), min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start: paragraph_end] + [102]
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            # print(f"question:{self.tokenizer.decode(tokenized_question.ids[:self.max_question_len])}, answer:{self.tokenizer.decode((input_ids_question + input_ids_paragraph)[answer_start_token: answer_end_token + 1])}")
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token
        else:
            # Validation/Testing
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]
                # print(f"question:{self.tokenizer.decode(input_ids_question).replace(' ', '')}, answer:{self.tokenizer.decode(input_ids_paragraph).replace(' ', '')}")
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # 如果seq的长度比最大长度小，用0填充
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # xlm-robert-large模型上，需要全部设置位0，为什么？不清楚
        # bert-base-chinese模型上，paragraph部分设置为1
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask
