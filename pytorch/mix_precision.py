import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.linear = nn.Sequential(
            *(nn.Linear(10000, 10000) for _ in range(10))
        )

    def forward(self, x):
        return self.linear(x)


def test_fp32():
    model = Layer()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.randn(1, 10000)
    x = data.cuda()
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        memory = torch.cuda.max_memory_allocated()
        print(f'step memory allocate: {memory / 1e9:.3f}G')


def test_fp16():
    torch.cuda.init()
    model = Layer().cuda()
    scaler = torch.cuda.amp.GradScaler()
    optimizer =  torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.randn(1, 10000)
    x = data.cuda()
    for _ in range(10):
        with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            output = model(x)
            loss = output.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        memory = torch.cuda.max_memory_allocated()
        print(f'memory allocated: {memory / 1e9:.3f}G')


if __name__ == '__main__':
    test_fp32()
    test_fp16()
