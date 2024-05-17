import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
EPOCH = 1
IMG_SIZE = 256
BATCH_SIZE= 6
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


base_path = 'D:\\python\\12203030118陈珊珊\\人工智能导论设计'  # 注意双反斜杠用于转义
data_dir = os.path.join(base_path, 'data_dog_cat')
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')


classes_name = os.listdir(train_path)

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])


class DogDataset(Dataset):
    def __init__(self, paths, classes_name, transform=None):
        self.paths = self.make_path(paths, classes_name)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.paths[idx].split(';')[0]
        img = Image.open(image)
        label = self.paths[idx].split(';')[1]
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def make_path(self, path, classes_name):
        # path: ./data1_dog_cat/train
        # path = './data1_dog_cat/train'
        path_list = []

        for class_name in classes_name:
            names = os.listdir(path + '/' + class_name)
            for name in names:
                p = os.path.join(path + '/' + class_name, name)
                label = str(classes_name.index(class_name))
                path_list.append(p + ';' + label)
        return path_list


train_dataset = DogDataset(train_path, classes_name, train_transforms)
val_dataset = DogDataset(test_path, classes_name, val_transforms)
image_dataset = {'train': train_dataset, 'valid': val_dataset}

image_dataloader = {x: DataLoader(image_dataset[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_dataset[x]) for x in ['train', 'valid']}
def conv_bn(in_channels,out_channels,kernel_size, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride,
                  padding=kernel_size // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class GlobalAveragePool2D():
    def __init__(self, keepdim=True):
        self.keepdim = keepdim

    def __call__(self, inputs):
        return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)


class SSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSEBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.globalAvgPool = GlobalAveragePool2D()
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        bn = self.norm(inputs)
        x = self.globalAvgPool(bn)
        x = self.conv(x)
        x = self.sigmoid(x)

        z = torch.mul(bn, x)
        return z

class Downsampling_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling_block, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = conv_bn(self.in_channels, self.out_channels, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.globalAvgPool = GlobalAveragePool2D()
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)

        y = self.conv2(inputs)

        z = self.globalAvgPool(inputs)
        z = self.conv3(z)
        z = self.sigmoid(z)

        a = x + y
        b = torch.mul(a, z)
        out = self.act(b)
        return out

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = 2 * self.in_channels
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = conv_bn(self.mid_channels, self.out_channels, kernel_size=1, stride=1, groups=2)
        self.conv2 = conv_bn(self.mid_channels, self.out_channels, kernel_size=3, stride=2, groups=2)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=1, groups=2)
        self.globalAvgPool = GlobalAveragePool2D()
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.group = in_channels

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, input1, input2):

        a = torch.cat([self.bn(input1), self.bn(input2)], dim=1)

        a = self.channel_shuffle(a)

        x = self.avgpool(a)

        x = self.conv1(x)

        y = self.conv2(a)

        z = self.globalAvgPool(a)

        z = self.conv3(z)
        z = self.sigmoid(z)

        a = x + y

        b = torch.mul(a, z)
        out = self.act(b)
        return out

class Stream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sse = nn.Sequential(SSEBlock(self.in_channels, self.out_channels))
        self.fuse = nn.Sequential(FuseBlock(self.in_channels, self.out_channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs):
        a = self.sse(inputs)
        b = self.fuse(inputs)
        c = a + b

        d = self.act(c)
        return d


class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = conv_bn(self.in_channels, self.out_channels, kernel_size=3, stride=1)

    def forward(self, inputs):
        a = self.conv1(inputs)
        b = self.conv2(inputs)

        c = a + b
        return c


class ParNetEncoder(nn.Module):
    def __init__(self, in_channels, block_channels, depth) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.depth = depth
        self.d1 = Downsampling_block(self.in_channels, self.block_channels[0])
        self.d2 = Downsampling_block(self.block_channels[0], self.block_channels[1])
        self.d3 = Downsampling_block(self.block_channels[1], self.block_channels[2])
        self.d4 = Downsampling_block(self.block_channels[2], self.block_channels[3])
        self.d5 = Downsampling_block(self.block_channels[3], self.block_channels[4])
        self.stream1 = nn.Sequential(
            *[Stream(self.block_channels[1], self.block_channels[1]) for _ in range(self.depth[0])]
        )

        self.stream1_downsample = Downsampling_block(self.block_channels[1], self.block_channels[2])

        self.stream2 = nn.Sequential(
            *[Stream(self.block_channels[2], self.block_channels[2]) for _ in range(self.depth[1])]
        )

        self.stream3 = nn.Sequential(
            *[Stream(self.block_channels[3], self.block_channels[3]) for _ in range(self.depth[2])]
        )

        self.stream2_fusion = Fusion(self.block_channels[2], self.block_channels[3])
        self.stream3_fusion = Fusion(self.block_channels[3], self.block_channels[3])

    def forward(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)

        y = self.stream1(x)
        y = self.stream1_downsample(y)

        x = self.d3(x)

        z = self.stream2(x)
        z = self.stream2_fusion(y, z)

        x = self.d4(x)

        a = self.stream3(x)
        b = self.stream3_fusion(z, a)

        x = self.d5(b)
        return x


class ParNetDecoder(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return self.softmax(x)


class ParNet(nn.Module):
    def __init__(self, in_channels, n_classes, block_channels=[64, 128, 256, 512, 2048], depth=[4, 5, 5]) -> None:
        super().__init__()
        self.encoder = ParNetEncoder(in_channels, block_channels, depth)
        self.decoder = ParNetDecoder(block_channels[-1], n_classes)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)

        return x

def parnet_s(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_channels=[64, 96, 192, 384, 1280])


def parnet_m(in_channels, n_classes):
    model = ParNet(in_channels, n_classes, block_channels=[64, 128, 256, 512, 2048])
    return model


def parnet_l(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_channels=[64, 160, 320, 640, 2560])


def parnet_xl(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_channels=[64, 200, 400, 800, 3200])

model_ft = parnet_s(3, len(classes_name))
model_ft.to(DEVICE)
print(model_ft)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-3)#指定 新加的fc层的学习率

cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_accuracy = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print_loss = loss.data.item()
        sum_loss += print_loss
        accuracy = torch.mean((torch.argmax(F.softmax(output, dim=-1), dim=-1) == target).type(torch.FloatTensor))
        total_accuracy += accuracy.item()
        if (batch_idx + 1) % 10 == 0:
            ave_loss = sum_loss / (batch_idx + 1)
            acc = total_accuracy / (batch_idx + 1)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))

            print('epoch:%d,loss:%.4f,train_acc:%.4f' % (epoch, ave_loss, acc))


ACC = 0


# 验证过程
def val(model, device, test_loader):
    global ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        if acc > ACC:
            torch.save(model_ft, 'model_' + 'epoch_' + str(epoch) + '_' + 'ACC-' + str(round(acc, 3)) + '.pth')
            ACC = acc


# 训练

for epoch in range(1, EPOCH + 1):
    train(model_ft, DEVICE, image_dataloader['train'], optimizer, epoch)
    cosine_schedule.step()
    val(model_ft, DEVICE, image_dataloader['valid'])


import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
import os

classes = ['cat', 'dog']
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('模型加载中!!!!!!!!!')


model_path = os.path.join(base_path, 'model_epoch_1_ACC-0.548.pth')

model = torch.load(model_path)
# 模型路径
print('模型加载成功!!!!!!!!!')
model.eval()
model.to(DEVICE)

# 预测图片路径

img_path = 'D:\\python\\12203030118陈珊珊\\人工智能导论设计\\data_dog_cat\\test\\dog'
# 使用 os.path.join 来确保路径是正确的，并且自动处理分隔符
path = os.path.join(img_path,  'dog.10014.jpg')


img = Image.open(path)
image = transform_test(img)
image.unsqueeze_(0)
image = Variable(image).to(DEVICE)
out = model(image)
_, pred = torch.max(out.data, 1)
# 在图上显示预测结果
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 30)  # 设置字体
content = classes[pred.data.item()]
draw.text((0, 0), content, font=font)
img.show()


