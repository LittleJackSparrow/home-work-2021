import torch
from tqdm.auto import tqdm


def evaluate(data, output, tokenizer):
    """

    :param data: data当中包含input_ids，token_type_ids，attention_mask_ids，取data[0]就是数据的IDs
                 另外，验证的时候data是一个问题对应N个文本的数据，因此会有data[0].shape[1]个windows
    :param output:
    :param tokenizer:
    :return:
    """
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # max函数是按照dim=0的维度，对结果进行softmax函数计算，得到最大的可能和可能的索引
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        prob = start_prob + end_prob
        if (end_index > start_index) & (prob > max_prob):
            max_prob = prob
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
    return answer.replace(' ', '')


def train(model, num_epoch, train_loader, dev_loader, device, accelerator, optimizer, logging_step, dev_questions, validation=True, model_save_dir=None, tokenizer=None, scheduler=None):
    model.train()
    print("Start Training ...")
    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0
        accum_iter = 4
        # for data in tqdm(train_loader):
        for i, data in enumerate(tqdm(train_loader)):
            with torch.set_grad_enabled(True):
                # Load all data into GPU
                data = [i.to(device) for i in data]
                # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
                # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
                # 前向传播，输出位QuestionAnsweringModelOutput对象
                # start_logits->起始位置的预测分数   end_logits->结束位置的预测分数
                # loss->模型的损失值，损失函数默认是交叉熵，暂时不知道如何自定义
                output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
                # Choose the most probable start position / end position
                # 过softmax函数得到结果，因为output.start_logits得到的是16行，193列的数据，而softmax函数针对的是每一行中元素的最大值，索引dim=1
                # 如果是dim=0，得到的就是193个了
                start_index = torch.argmax(output.start_logits, dim=1)
                end_index = torch.argmax(output.end_logits, dim=1)
                # Prediction is correct only if both start_index and end_index are correct
                train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
                # 计算损失，看不懂，据说是可以提升梯度计算的稳定性
                output.loss /= accum_iter
                train_loss += output.loss
                # 反向传播
                if accelerator is None:
                    output.loss.backward()
                else:
                    accelerator.backward(output.loss)
                if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                    # 更新参数
                    optimizer.step()
                    # 梯度置空
                    optimizer.zero_grad()
                    ##### TODO: Apply linear learning rate decay #####
                    scheduler.step()
                step += 1
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0
        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            # 不更新梯度
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    # squeeze()方法：将维度为1的轴去除
                    output = model(input_ids=data[0].squeeze().to(device), token_type_ids=data[1].squeeze().to(device), attention_mask=data[2].squeeze().to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, output, tokenizer) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            model.train()
    # Save a model and its configuration file to the directory 「saved_model」
    # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
    print("Saving Model ...")
    model.save_pretrained(model_save_dir)


def test(model, test_loader, device, test_questions, tokenizer):
    print("Evaluating Test Set ...")
    result = []
    # 开启验证模式
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device), attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output, tokenizer))

    result_file = "result.csv"
    with open(result_file, 'w') as f:
        f.write("question_text,answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['question_text']},{result[i].replace(',', '')}\n")
    print(f"Completed! Result is in {result_file}")
