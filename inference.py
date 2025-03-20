import torch
import albumentations as albu
import numpy as np
import os,glob
from torch.utils import data
from tqdm import tqdm
from models import sam_feat_seg_model_registry
from test_dataset import Dataset
import matplotlib.pyplot as plt
from PIl import Image

class Config(object):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    img_path = r'/media/fmost/student/zhangshilong/sam2d-dataset/soma/test/imgs'
    img_out_path_prompt = r'/media/share01/public/zsl/BrainScope-SAM/scripts/hot_graph'
    model_path = r'/media/share01/public/zsl/BrainScope-SAM/scripts/all_model_best-0807.pth'
    img_type = 'tif'
    model_type = 'vit_l'
    img_size = 1024
    num_classes = 2
    batch_size = 1

def test():
    configs = Config()
    suffix = '*.' + configs.img_type
    img_names = glob.glob(os.path.join(configs.img_path, suffix))

    test_transform = albu.Compose([
        albu.Resize(1024, 1024),
        albu.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
    ],is_check_shapes=False)
    test_set = Dataset(configs=configs, names=img_names, transform=test_transform)
    test_loader = data.DataLoader(test_set, batch_size=configs.batch_size, shuffle=True, drop_last=True)

    model = sam_feat_seg_model_registry[configs.model_type](num_classes=configs.num_classes, checkpoint=configs.model_path, img_size=configs.img_size, iter_2stage=1)

    model.to(configs.device)

    model.eval()

    with torch.no_grad():
        for img, name in tqdm(test_loader, total=len(test_loader)):
            img = img.to(configs.device)
            img_out_initial, img_out_end, prompt_embedding = model(img)

            #可视化提示编码的位置显著性
            prompt_embedding1 = prompt_embedding[:, 1, :, :] - prompt_embedding[:, 0, :, :]

            prompt_embedding1 = prompt_embedding1.view(prompt_embedding1.shape[1], prompt_embedding1.shape[2])

            feature = prompt_embedding1.cpu().data.numpy()


            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))
            # to [0,255]
            feature = np.round(feature * 255)

            plt.imshow(feature, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.axis('off')  # Disable the axis
            plt.savefig(configs.img_out_path_prompt + '/' + name['img_id'][0])
            plt.close()

            #可视化参数量与算力
            #flops, params = profile(model, inputs=(img,))
            #print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
            #print("Params=", str(params / 1e6) + '{}'.format("M"))

            pred_intial = img_out_initial.argmax(dim=1).unsqueeze(dim=1)
            pred_end = img_out_end.argmax(dim=1).unsqueeze(dim=1)
            for i in range(img.size(0)):

                img_end = pred_end.detach()[i].cpu().numpy() # *255
                img_end = Image.fromarray(img_end.astype(np.uint8).squeeze(axis=0))
                # print(type(img_end))
                img_end = img_end.resize((Config.img_size, Config.img_size), Image.NEAREST)

                img_end.save(configs.img_out_path + '/' + name['img_id'][i])


if __name__=='__main__':
    test()

    