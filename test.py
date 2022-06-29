import jieba
# from d2l import torch as d2l
# batch_size, num_steps = 64, 10
# train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# text = d2l.preprocess_nmt(d2l.read_data_nmt())
# source, target = d2l.tokenize_nmt(text, 600)

# timer = d2l.Timer()
# metric = d2l.Accumulator(2) # 训练损失总和，词元数量
# d2l.grad_clipping(net, 1)

sent = "太贵了"
words = jieba.lcut(sent.strip())
print(words)