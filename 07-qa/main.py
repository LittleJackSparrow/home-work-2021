import matplotlib.pyplot as plt
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering

import train_model
import utils.settings as settings
from dataset import questions_paragraphs_tokenized as tokenized, get_data_loader

"""
    python=3.11.5   pytorch=2.1.0  transformers=4.32.1
    accelerate=0.26.1 一个针对pytorch和深度学习的加速库
"""


def train_val(model, train_loader, accelerator):
    learning_rate = 1e-4
    num_epoch = 1
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=160, num_training_steps=num_epoch * len(train_loader))
    model, optimizer, train_loader = settings.prepare(accelerator, model, optimizer, train_loader)
    train_model.train(model, num_epoch, train_loader, dev_loader, device, accelerator, optimizer, 100,
                      dev_questions, True, model_save_dir, tokenizer, scheduler)


def test():
    map_location = 'cpu'
    # 加载已经保存的参数
    model_params = torch.load(model_save_dir + '/pytorch_model.bin', map_location)  # Load your best model
    # 使用加载的参数更新模型的状态字典
    model.load_state_dict(model_params)
    train_model.test(model, test_loader, device, test_questions, tokenizer)


def plot_lr(model, train_loader):
    learning_rate = 1e-4
    num_epoch = 1
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=160, num_training_steps=num_epoch * len(train_loader))
    step = []
    lr_ = []
    print(num_epoch * len(train_loader))
    for x in range(num_epoch * len(train_loader)):
        scheduler.step()
        step.append(x + 1)
        lr_.append(optimizer.param_groups[0]['lr'])
    plt.plot(step, lr_)
    plt.show()


def random_positional():
    import random
    answer_start_token = 350
    answer_end_token = 360
    max_paragraph_len = 150
    tokenized_paragraph = 360
    # answer_end_token - max_paragraph_len
    start_range = random.randint(max(0, answer_end_token - max_paragraph_len), answer_start_token)
    min(start_range, tokenized_paragraph - max_paragraph_len)


def bert_base_chinese_model():
    """
        epoch为2时计算acc在0.76左右
        epoch提升时，在训练集上的出现了过拟合，能达到0.95，但是在测试集上面并没有得到提升，还是在0.76左右
    :return:
    """
    # 指定预训练模型的路径
    model_path = "D:/01-workspace/github/dataset/07-qa/model/bert-base-chinese"
    # 加载分词器
    bert_base_tokenizer = BertTokenizerFast.from_pretrained(model_path)
    # 加载模型
    bert_base_model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    return bert_base_tokenizer, bert_base_model


def xlm_roberta_large_model():
    """
        在使用了混合精度和梯度累积之后，模型在12G显存的机器上仍旧是内存溢出，无法验证
        可以想办法使用小点的模型
    :return:
    """
    # 指定预训练模型的路径
    model_path = "../model/xlm-roberta-large"
    # 加载分词器
    xlm_roberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)
    # print(f'{xlm_roberta_tokenizer.vocab_size}--->{config.vocab_size}')
    # 加载模型
    xlm_roberta_model = XLMRobertaForQuestionAnswering.from_pretrained(model_path).to(device)
    return xlm_roberta_tokenizer, xlm_roberta_model


if __name__ == '__main__':
    settings.same_seeds(0)
    accelerator = settings.get_accelerate(True)
    device = settings.get_device(accelerator)

    tokenizer, model = bert_base_chinese_model()
    root = "D:/01-workspace/github/dataset/07-qa"
    train_questions, train_questions_tokenized, train_paragraphs_tokenized = tokenized(tokenizer, f'{root}/hw7_train.json')
    dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized = tokenized(tokenizer, f'{root}/hw7_dev.json')
    test_questions, test_questions_tokenized, test_paragraphs_tokenized = tokenized(tokenizer, f'{root}/hw7_test.json')

    train_loader = get_data_loader('train', batch_size=16, questions=train_questions, questions_tokenized=train_questions_tokenized, paragraphs_tokenized=train_paragraphs_tokenized, tokenizer=tokenizer)
    dev_loader = get_data_loader('dev', batch_size=1, questions=dev_questions, questions_tokenized=dev_questions_tokenized, paragraphs_tokenized=dev_paragraphs_tokenized, tokenizer=tokenizer, shuffle=False)
    test_loader = get_data_loader('test', batch_size=1, questions=test_questions, questions_tokenized=test_questions_tokenized, paragraphs_tokenized=test_paragraphs_tokenized, tokenizer=tokenizer, shuffle=False)
    model_save_dir = "D:/01-workspace/github/dataset/07-qa/model"



    train_val(model, train_loader, accelerator)
    # test()
    # plot_lr(model, train_loader)
    # random_positional()
