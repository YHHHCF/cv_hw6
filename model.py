import torch
import torch.nn as nn
import torchvision.models as models
import cv2


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layers_1 = nn.Sequential(*list(model.children())[:-1])
        self.layers_2 = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5),
                                      nn.Linear(in_features=4096, out_features=4096, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5))
        self.layers_3 = nn.Linear(in_features=4096, out_features=1000, bias=True)
        print(model)

    def forward(self, x):
        x = self.layers_1(x)
        x = x.reshape(x.shape[0], -1)
        emb = self.layers_2(x)
        out = self.layers_3(emb)

        return out, emb


def get_model():
    pt_model = models.vgg11(pretrained=True)

    new_model = Net(pt_model)
    params_pt = pt_model.state_dict()
    params_new = new_model.state_dict()

    params_list = []
    for pa in params_pt:
        params_list.append(params_pt[pa])

    idx = 0
    for pa in params_new:
        params_new[pa] = params_list[idx]
        idx += 1

    new_model.load_state_dict(params_new)

    return new_model


def print_model(model):
    params = model.state_dict()
    for key in params.keys():
        print(key, torch.max(params[key]))


def save_ckpt(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    return


if __name__ == '__main__':
    new_model = get_model()
    path = './data/pf_desk.jpg'
    image = cv2.imread(path)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.reshape((1, 3, image.shape[0], image.shape[1]))
    print(image.shape)
    out, emb = new_model(image)
    print(torch.argmax(out, dim=1))
    print(emb.shape)
