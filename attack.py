# attack.py
import torch
import torch.nn.functional as F

def fgsm_attack(model, inputs, labels, epsilon=0.01):
    """基于嵌入向量的FGSM攻击（适配修改后的BertEncoder）"""
    model.train()  # 保持训练模式以获取梯度
    attention_mask = inputs['attention_mask']
    h_index = inputs['h_index']
    t_index = inputs['t_index']

    # 1. 获取原始嵌入向量（开启梯度追踪）
    embedding, _, _ = model(
        input_ids=inputs['input_ids'],
        attention_mask=attention_mask,
        h_index=h_index,
        t_index=t_index,
        return_embedding=True  # 利用修改后的接口获取嵌入向量
    )
    embedding = embedding.detach().requires_grad_(True)  # 嵌入向量需梯度

    # 2. 基于嵌入向量计算损失（用于求梯度）
    _, _, feature = model(
        attention_mask=attention_mask,
        h_index=h_index,
        t_index=t_index,
        inputs_embeds=embedding  # 传入带梯度的嵌入向量
    )
    # 计算分类损失（需与模型输出匹配，此处假设用对比损失）
    loss = F.cross_entropy(model.classifier(feature), labels)
    model.zero_grad()
    loss.backward()  # 求嵌入向量的梯度

    # 3. 生成扰动嵌入并返回
    perturbed_embedding = embedding + epsilon * embedding.grad.sign()
    return {
        'inputs_embeds': perturbed_embedding.detach(),  # 返回扰动后的嵌入
        'attention_mask': attention_mask,
        'h_index': h_index,
        't_index': t_index
    }
