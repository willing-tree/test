# encoding:utf-8
import os
import nni
import math
import time
import json
import torch
import argparse
import torch.nn.functional as F
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
#from transformers import BertTokenizer, BertModel
from transformers import AutoConfig, AutoTokenizer, AutoModel

from nni.utils import merge_parameter

from model import BertEncoder, Classifier
from data_process import FewRelProcessor, tacredProcessor
from utils import collate_fn, save_checkpoint, get_prototypes, memory_select, set_random_seed, compute_cos_sim, get_augmentative_data

import copy
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_logger(args):
    """创建并配置日志记录器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if args.log_path:
            file_handler = logging.FileHandler(args.log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"
red_print = "\033[1;31;40m"  # 红色打印攻击结果


def setup_logging(log_dir):

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir),  # 将日志写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger()


def do_train(args, tokenizer, processor, i_exp):
    """完整训练流程，包含攻击评估与性能对比"""
    # 初始化日志和设备
    logger = get_logger(args)
    device = args.device
    blue_print = "\033[1;34;40m"
    green_print = "\033[1;32;40m"
    red_print = "\033[1;31;40m"
    default_print = "\033[0m"

    # 加载数据处理器和关系映射
    rel2id = processor.get_rel2id()
    num_relations = len(rel2id)
    logger.info(f"Total relations: {num_relations}")

    # 初始化模型组件
    current_encoder = BertEncoder(args, tokenizer, encode_style=args.encode_style)
    current_classifier = Classifier(args, num_relations=0)  # 初始分类器为空
    prev_encoder = None
    prev_classifier = None
    memory_data = None
    proto_features = None  # 关系原型特征

    # 记录准确率的列表（正常/攻击后）
    task_acc = []  # 每个任务的准确率
    memory_acc = []  # 跨任务累积准确率

    # 按任务顺序训练
    for i in range(args.task_num):
        logger.info(f"{blue_print}===== Training Task {i} =====")

        # 1. 加载当前任务数据
        traindata, testdata = processor.get(i)
        train_len = processor.get_len(i)
        logger.info(f"Task {i} - Train samples: {len(traindata)}, Test samples: {len(testdata)}")

        # 2. 初始化当前任务的分类器（增量扩展）
        current_classifier.incremental_learning(train_len)
        current_classifier.to(device)

        # 3. 训练当前任务
        logger.info(f"Training on task {i}...")
        train_val_task(
            args, current_encoder, current_classifier, traindata,
            tokenizer, rel2id, i, prev_encoder, prev_classifier, memory_data
        )

        # 4. 生成当前任务的关系原型
        current_prototypes, current_proto_features = get_prototypes(args, current_encoder, traindata, train_len)
        if i == 0:
            prototypes = current_prototypes
            proto_features = current_proto_features
        else:
            prototypes = torch.cat([prototypes, current_prototypes], dim=0)
            proto_features = torch.cat([proto_features, current_proto_features], dim=0)

        # 5. 评估当前任务（正常+攻击）
        # 5.1 任务内评估
        acc_clean = evaluate(args, current_encoder, current_classifier, testdata, rel2id)
        if args.attack_method is not None:
            acc_attack = evaluate(
                args, current_encoder, current_classifier, testdata, rel2id,
                attack_method=args.attack_method, epsilon=args.epsilon
            )
            logger.info(
                f"{blue_print}Task {i} Accuracy - Clean: {acc_clean:.4f}, Attacked: {acc_attack:.4f}, Drop: {acc_clean - acc_attack:.4f}{default_print}")
            task_acc.append((acc_clean, acc_attack))
        else:
            logger.info(f"{blue_print}Task {i} Accuracy: {acc_clean:.4f}{default_print}")
            task_acc.append(acc_clean)

        # 5.2 跨任务累积评估（所有已训练任务）
        testset = processor.get_testset(i)  # 累积测试集（任务0-i）
        if prev_encoder is not None:
            acc_clean = evaluate(args, current_encoder, current_classifier, testset, rel2id, proto_features)
        else:
            acc_clean = evaluate(args, current_encoder, current_classifier, testset, rel2id)

        if args.attack_method is not None:
            if prev_encoder is not None:
                acc_attack = evaluate(
                    args, current_encoder, current_classifier, testset, rel2id, proto_features,
                    attack_method=args.attack_method, epsilon=args.epsilon
                )
            else:
                acc_attack = evaluate(
                    args, current_encoder, current_classifier, testset, rel2id,
                    attack_method=args.attack_method, epsilon=args.epsilon
                )
            logger.info(
                f"{green_print}Cumulative Accuracy (0-{i}) - Clean: {acc_clean:.4f}, Attacked: {acc_attack:.4f}, Drop: {acc_clean - acc_attack:.4f}{default_print}")
            memory_acc.append((acc_clean, acc_attack))
        else:
            logger.info(f"{green_print}Cumulative Accuracy (0-{i}): {acc_clean:.4f}{default_print}")
            memory_acc.append(acc_clean)

        # 6. 记忆样本选择与更新
        if args.memory_size > 0:
            # 从当前任务选择记忆样本
            new_memory = memory_select(args, traindata, current_encoder, current_classifier, train_len)
            memory_data = new_memory if memory_data is None else memory_data + new_memory
            # 用记忆样本进行抗遗忘训练
            logger.info(f"Training on memory data (size: {len(memory_data)})...")
            train_val_memory(
                args, current_encoder, current_classifier, memory_data,
                tokenizer, rel2id, i, proto_features
            )

        # 7. 保存当前模型作为下一轮的前置模型
        prev_encoder = copy.deepcopy(current_encoder)
        prev_classifier = copy.deepcopy(current_classifier)
        torch.cuda.empty_cache()

    # 8. 最终结果统计
    logger.info(f"{red_print}===== Final Results =====")
    if args.attack_method is not None:
        # 任务内准确率统计
        task_clean = [x[0] for x in task_acc]
        task_attack = [x[1] for x in task_acc]
        avg_task_clean = sum(task_clean) / len(task_clean)
        avg_task_attack = sum(task_attack) / len(task_attack)
        logger.info(
            f"Task Avg - Clean: {avg_task_clean:.4f}, Attacked: {avg_task_attack:.4f}, Avg Drop: {avg_task_clean - avg_task_attack:.4f}")

        # 跨任务准确率统计
        memory_clean = [x[0] for x in memory_acc]
        memory_attack = [x[1] for x in memory_acc]
        avg_memory_clean = sum(memory_clean) / len(memory_clean)
        avg_memory_attack = sum(memory_attack) / len(memory_attack)
        logger.info(
            f"Cumulative Avg - Clean: {avg_memory_clean:.4f}, Attacked: {avg_memory_attack:.4f}, Avg Drop: {avg_memory_clean - avg_memory_attack:.4f}")
    else:
        avg_task = sum(task_acc) / len(task_acc) if task_acc else 0
        avg_memory = sum(memory_acc) / len(memory_acc) if memory_acc else 0
        logger.info(f"Average Task Accuracy: {avg_task:.4f}")
        logger.info(f"Average Cumulative Accuracy: {avg_memory:.4f}")

    return task_acc, memory_acc


def train_val_task(args, encoder, classifier, traindata, valdata, rel2id, train_len):
    logger = setup_logging(args.log_dir)
    dataloader = DataLoader(traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)

    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': classifier.parameters(), 'lr': args.classifier_lr}
        ], eps=args.adam_epsilon)
    # todo add different learning rate for each layer

    best_acc = 0.0
    for epoch in range(args.epoch_num_task):
        encoder.train()
        classifier.train()
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            hidden, _ = encoder(**inputs)

            inputs = {
                'hidden': hidden,
                'labels': batch[4].to(args.device)
            }
            loss, _ = classifier(**inputs)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()



    acc = evaluate(args, encoder, classifier, valdata, rel2id)
    best_acc = max(acc, best_acc)

    # 将进度条信息写入日志文件
    logger.info(f"Epoch {epoch + 1}/{args.epoch_num_task}, Step {step + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    logger.info(f'Evaluate on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
    return encoder


def train_val_memory(args, model, prev_model, traindata, aug_traindata, testdata, rel2id, memory_len, aug_memory_len, prototypes, proto_features, task_prototypes, task_proto_features):
    logger = setup_logging(args.log_dir)
    enc, cls = model
    prev_enc, prev_cls = prev_model
    dataloader = DataLoader(aug_traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)

    optimizer = AdamW([
        {'params': enc.parameters(), 'lr': args.encoder_lr},
        {'params': cls.parameters(), 'lr': args.classifier_lr}
        ], eps=args.adam_epsilon)

    prev_enc.eval()
    prev_cls.eval()
    best_acc = 0.0
    for epoch in range(args.epoch_num_memory):
        enc.train()
        cls.train()
        for step, batch in enumerate(tqdm(dataloader)):
            enc_inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            hidden, feature = enc(**enc_inputs)
            with torch.no_grad():
                prev_hidden, prev_feature = prev_enc(**enc_inputs)

            labels = batch[4].to(args.device)
            cont_loss = contrastive_loss(args, feature, labels, prototypes, proto_features, prev_feature)
            cont_loss.backward(retain_graph=True)

            rep_loss = replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels, prototypes, proto_features)
            rep_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            acc = evaluate(args, enc, cls, testdata, rel2id, proto_features)
            best_acc = max(best_acc, acc)
            logger.info(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
            nni.report_intermediate_result(acc)

            prototypes_replay, proto_features_replay = get_prototypes(args, enc, traindata, memory_len)
            prototypes, proto_features = (1-args.beta)*task_prototypes + args.beta*prototypes_replay, (1-args.beta)*task_proto_features + args.beta*proto_features_replay
            prototypes = F.layer_norm(prototypes, [args.hidden_dim])
            proto_features = F.normalize(proto_features, p=2, dim=1)

    return enc


def contrastive_loss(args, feature, labels, prototypes, proto_features=None, prev_feature=None):
    # supervised contrastive learning loss
    dot_div_temp = torch.mm(feature, proto_features.T) / args.cl_temp # [batch_size, rel_num]
    dot_div_temp_norm = dot_div_temp - 1.0 / args.cl_temp
    exp_dot_temp = torch.exp(dot_div_temp_norm) + 1e-8 # avoid log(0)

    mask = torch.zeros_like(exp_dot_temp).to(args.device)
    mask.scatter_(1, labels.unsqueeze(1), 1.0)
    cardinalities = torch.sum(mask, dim=1)

    log_prob = -torch.log(exp_dot_temp / torch.sum(exp_dot_temp, dim=1, keepdim=True))
    scloss_per_sample = torch.sum(log_prob*mask, dim=1) / cardinalities
    scloss = torch.mean(scloss_per_sample)
    
    # focal knowledge distillation loss
    if prev_feature is not None:
        with torch.no_grad():
            prev_proto_features = proto_features[:proto_features.shape[1]-args.relnum_per_task]
            prev_sim = F.softmax(torch.mm(feature, prev_proto_features.T) / args.cl_temp / args.kd_temp, dim=1)

            prob = F.softmax(torch.mm(feature, proto_features.T) / args.cl_temp / args.kd_temp, dim=1)
            focal_weight = 1.0 - torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
            focal_weight = focal_weight ** args.gamma

            target = F.softmax(torch.mm(prev_feature, prev_proto_features.T) / args.cl_temp, dim=1) # [batch_size, prev_rel_num]

        source = F.log_softmax(torch.mm(feature, prev_proto_features.T) / args.cl_temp, dim=1) # [batch_size, prev_rel_num]
        target = target * prev_sim + 1e-8
        fkdloss = torch.sum(-source * target, dim=1)
        fkdloss = torch.mean(fkdloss * focal_weight)
    else:
        fkdloss = 0.0
    
    # margin loss
    if proto_features is not None:
        with torch.no_grad():
            sim = torch.mm(feature, proto_features.T)
            neg_sim = torch.scatter(sim, 1, labels.unsqueeze(1), -10.0)
            neg_indices = torch.argmax(neg_sim, dim=1)
        
        pos_proto = proto_features[labels]
        neg_proto = proto_features[neg_indices]

        positive = torch.sum(feature * pos_proto, dim=1)
        negative = torch.sum(feature * neg_proto, dim=1)

        marginloss = torch.maximum(args.margin - positive + negative, torch.zeros_like(positive).to(args.device))
        marginloss = torch.mean(marginloss)
    else:
        marginloss = 0.0

    loss = scloss + args.cl_lambda*marginloss + args.kd_lambda2*fkdloss
    return loss


def replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels, prototypes=None, proto_features=None):
    # cross entropy
    celoss, logits = cls(hidden, labels)
    with torch.no_grad():
        prev_logits, = prev_cls(prev_hidden)

    if prototypes is None:
        index = prev_logits.shape[1]
        source = F.log_softmax(logits[:, :index], dim=1)
        target = F.softmax(prev_logits, dim=1) + 1e-8
        kdloss = F.kl_div(source, target)
    else:
        # focal knowledge distillation
        with torch.no_grad():
            sim = compute_cos_sim(hidden, prototypes)
            prev_sim = sim[:, :prev_logits.shape[1]] # [batch_size, prev_rel_num]
            prev_sim = F.softmax(prev_sim / args.kd_temp, dim=1)

            prob = F.softmax(logits, dim=1)
            focal_weight = 1.0 - torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
            focal_weight = focal_weight ** args.gamma

        source = logits.narrow(1, 0, prev_logits.shape[1])
        source = F.log_softmax(source, dim=1)
        target = F.softmax(prev_logits, dim=1)
        target = target * prev_sim + 1e-8
        kdloss = torch.sum(-source * target, dim=1)
        kdloss = torch.mean(kdloss * focal_weight)
    
    rep_loss = celoss + args.kd_lambda1*kdloss
    return rep_loss


# main.py
def evaluate(args, model, classifier, valdata, rel2id, proto_features=None, attack_method=None, epsilon=0.01):
    model.eval()
    dataloader = DataLoader(valdata, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    pred_clean, pred_attack, golden = [], [], []  # 分别存储正常/攻击/真实标签

    for batch in tqdm(dataloader):
        # 原始输入
        clean_inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }
        labels = batch[4]
        golden.extend(labels.tolist())

        # 1. 正常评估（无攻击）
        with torch.no_grad():
            hidden_clean, feature_clean = model(** clean_inputs)
            logits_clean = classifier(hidden_clean)[0]
            prob_clean = F.softmax(logits_clean, dim=1)
            if proto_features is not None:
                logits_ncm = torch.mm(feature_clean, proto_features.T) / args.cl_temp
                prob_ncm = F.softmax(logits_ncm, dim=1)
                final_prob_clean = args.alpha * prob_clean + (1 - args.alpha) * prob_ncm
            else:
                final_prob_clean = prob_clean
        pred_clean.extend(torch.argmax(final_prob_clean, dim=1).cpu().tolist())

        # 2. 攻击评估（若指定攻击方法）
        if attack_method is not None:
            from attack import fgsm_attack  # 导入攻击函数
            # 生成对抗样本（扰动嵌入向量）
            perturbed_inputs = fgsm_attack(model, clean_inputs, labels.to(args.device), epsilon=epsilon)
            with torch.no_grad():
                # 用扰动后的嵌入向量评估
                hidden_attack, feature_attack = model(**perturbed_inputs)
                logits_attack = classifier(hidden_attack)[0]
                prob_attack = F.softmax(logits_attack, dim=1)
                if proto_features is not None:
                    logits_ncm_attack = torch.mm(feature_attack, proto_features.T) / args.cl_temp
                    prob_ncm_attack = F.softmax(logits_ncm_attack, dim=1)
                    final_prob_attack = args.alpha * prob_attack + (1 - args.alpha) * prob_ncm_attack
                else:
                    final_prob_attack = prob_attack
            pred_attack.extend(torch.argmax(final_prob_attack, dim=1).cpu().tolist())

    # 计算准确率
    golden = torch.tensor(golden)
    acc_clean = (torch.tensor(pred_clean) == golden).float().mean().item()
    if attack_method is not None:
        acc_attack = (torch.tensor(pred_attack) == golden).float().mean().item()
        return acc_clean, acc_attack  # 返回正常/攻击后的准确率
    else:
        return acc_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoint", type=str)
    parser.add_argument("--dataset_name", default="tacred", type=str)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--cuda_device", default=1, type=int)

    parser.add_argument("--plm_name", default="bert-base-uncased", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--epoch_num_task", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--epoch_num_memory", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--hidden_dim", default=768 , type=int, help="Output dimension of encoder.")
    parser.add_argument("--feature_dim", default=64, type=int, help="Output dimension of projection head.")
    parser.add_argument("--encoder_lr", default=1e-5, type=float, help="The initial learning rate of encoder for AdamW.")
    parser.add_argument("--classifier_lr", default=1e-3, type=float, help="The initial learning rate of classifier for AdamW.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--alpha", default=0.6, type=float, help="Bagging Hyperparameter.")
    parser.add_argument("--beta", default=0.2, type=float, help="Prototype weight.")
    parser.add_argument("--cl_temp", default=0.1, type=float, help="Temperature for contrastive learning.")
    parser.add_argument("--cl_lambda", default=0.8, type=float, help="Hyperparameter for contrastive learning.")
    parser.add_argument("--margin", default=0.15, type=float, help="Hyperparameter for margin loss.")
    parser.add_argument("--kd_temp", default=0.5, type=float, help="Temperature for knowledge distillation.")
    parser.add_argument("--kd_lambda1", default=0.7, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--kd_lambda2", default=0.5, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--gamma", default=2.0, type=float, help="Hyperparameter of focal loss.")
    parser.add_argument("--encode_style", default="emarker", type=str, help="Encode style of encoder.")

    parser.add_argument("--experiment_num", default=5, type=int)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--set_task_order", default=True, type=bool)
    parser.add_argument("--read_from_task_order", default=True, type=bool)
    parser.add_argument("--task_num", default=10, type=int)
    parser.add_argument("--memory_size", default=10, type=int, help="Memory size for each relation.")
    parser.add_argument("--early_stop_patient", default=10, type=int)
    #增加log输出
    #parser.add_argument("--log_dir", type=str, default='',help="log.")

    #运行zwl
    parser.add_argument("--log_dir", type=str, default='./logs/train.log', help="Path to log file.")
    # 在parser.add_argument中添加
    parser.add_argument("--attack_method", default=None, type=str, help="攻击方法，如fgsm、pgd等，默认不攻击")
    parser.add_argument("--epsilon", default=0.01, type=float, help="攻击扰动强度")

    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    #增加log输出

    if args.cuda:
        device = "cuda:"+str(args.cuda_device)
    else:
        device = "cpu"

    args.device = device
    args.collate_fn = collate_fn

    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)


    tokenizer = AutoTokenizer.from_pretrained("./bert-based-uncased",additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    #tokenizer = BertTokenizer.from_pretrained(args.plm_name, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

    s = time.time()
    task_results, memory_results = [], []
    for i in range(args.experiment_num):
        set_random_seed(args)
        if args.dataset_name == "FewRel":
            processor = FewRelProcessor(args, tokenizer)
        else:
            processor = tacredProcessor(args, tokenizer)
        if args.set_task_order:
            processor.set_task_order("task_order.json", i)
        if args.read_from_task_order:
            processor.set_read_from_order(i)

        # 在main函数中，调用do_train后
        task_acc, memory_acc = do_train(args, tokenizer, processor, i)

        # 统计攻击相关结果
        if args.attack_method is not None:
            # 任务内准确率统计
            task_clean = [x[0] for x in task_acc]
            task_attack = [x[1] for x in task_acc]
            avg_task_clean = sum(task_clean) / len(task_clean)
            avg_task_attack = sum(task_attack) / len(task_attack)
            logger.info(
                f'{red_print}任务内平均准确率 - 正常: {avg_task_clean}, 攻击后: {avg_task_attack}, 平均下降: {avg_task_clean - avg_task_attack:.4f}{default_print}')

            # 跨任务准确率统计
            memory_clean = [x[0] for x in memory_acc]
            memory_attack = [x[1] for x in memory_acc]
            avg_memory_clean = sum(memory_clean) / len(memory_clean)
            avg_memory_attack = sum(memory_attack) / len(memory_attack)
            logger.info(
                f'{red_print}跨任务平均准确率 - 正常: {avg_memory_clean}, 攻击后: {avg_memory_attack}, 平均下降: {avg_memory_clean - avg_memory_attack:.4f}{default_print}')
        else:
            # 原统计逻辑（无攻击）
            avg_task = sum(task_acc) / len(task_acc)
            avg_memory = sum(memory_acc) / len(memory_acc)
            logger.info(f'Average task accuracy: {avg_task}, Average memory accuracy: {avg_memory}')
        # torch.cuda.empty_cache()
    e = time.time()

    task_results = torch.tensor(task_results, dtype=torch.float32)
    memory_results = torch.tensor(memory_results, dtype=torch.float32)
    logging(f'All task result: {task_results.tolist()}')
    logging(f'All memory result: {memory_results.tolist()}')

    task_results = torch.mean(task_results, dim=0).tolist()
    memory_results = torch.mean(memory_results, dim=0)
    final_average = torch.mean(memory_results).item()
    logging(f'Final task result: {task_results}')
    logging(f'Final memory result: {memory_results.tolist()}')
    logging(f'Final average: {final_average}')
    logging(f'Time cost: {e-s}s.')

    # 攻击后总平均结果（新增代码）
    if args.attack_method is not None:
        # 计算所有实验的攻击后平均准确率
        attack_memory_results = []
        for i in range(args.experiment_num):
            # 假设每个实验的攻击结果已记录，此处简化为重新评估
            # 实际需在do_train中保存attack_memory_acc并返回
            attack_acc = evaluate(...)  # 用最终模型评估测试集
            attack_memory_results.append(attack_acc)
        attack_final_average = torch.mean(torch.tensor(attack_memory_results)).item()
        print(
            f'{red_print}Attack final average: {attack_final_average}, total drop(下降): {final_average - attack_final_average:.4f}{default_print}')

    nni.report_final_result(final_average)

