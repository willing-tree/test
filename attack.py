import torch
import torch.nn.functional as F


def fgsm_attack(model, inputs, labels, epsilon=0.01):
    """
    FGSM攻击：对输入嵌入添加扰动
    model: 待攻击的模型（BertEncoder）
    inputs: 包含input_ids、attention_mask、h_index、t_index的字典
    labels: 样本标签（用于计算梯度）
    epsilon: 扰动强度
    """
    # 确保模型处于训练模式以获取梯度
    model.train()
    # 复制输入以避免修改原始数据
    input_ids = inputs['input_ids'].clone().detach().requires_grad_(True)
    attention_mask = inputs['attention_mask']
    h_index = inputs['h_index']
    t_index = inputs['t_index']

    # 前向传播计算损失
    hidden, _ = model(input_ids=input_ids, attention_mask=attention_mask, h_index=h_index, t_index=t_index)
    loss = F.cross_entropy(model.classifier(hidden)[0], labels)  # 假设classifier是模型的分类头

    # 反向传播计算输入梯度
    model.zero_grad()
    loss.backward()

    # 生成扰动（仅修改input_ids的嵌入，这里简化为直接修改input_ids的梯度方向）
    # 注意：实际文本攻击可能需要在嵌入空间添加扰动，而非直接修改token id
    perturbed_input_ids = input_ids + epsilon * input_ids.grad.sign()
    # 确保扰动后的token id仍在有效范围内（0~vocab_size）
    perturbed_input_ids = torch.clamp(perturbed_input_ids, min=0, max=model.tokenizer.vocab_size - 1).long()

    # 返回扰动后的输入
    return {
        'input_ids': perturbed_input_ids.detach(),
        'attention_mask': attention_mask,
        'h_index': h_index,
        't_index': t_index
    }

# 可添加其他攻击方法，如PGD、文本扰动（同义词替换）等