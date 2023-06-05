import os
import torch
import numpy as np

from PIL import Image
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import torchvision.transforms as standard_transforms
from network import *
from dataset.zurich_night_dataset import zurich_night_DataSet
from configs.test_config import get_arguments


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def main():
    input_path = "/home2/yijs/DANNet/data/dataset/Suncloud_mini_val_ref/"
    output_path = "/home2/yijs/DANNet/result/Suncloud_mini_val_ref_mask"

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda")


    args = get_arguments()
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes)
    if args.model == 'RefineNet':
        model = RefineNet(num_classes=args.num_classes, imagenet=False)
    
    input_path = args.input
    output_path = input_path.rstrip("/") + "_mask"
    print("input_path: ", input_path)
    print("output_path: ", output_path)
    os.makedirs(output_path, exist_ok=True)
    

    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    lightnet = LightNet()
    saved_state_dict = torch.load(args.restore_from_light)
    model_dict = lightnet.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    lightnet.load_state_dict(saved_state_dict)

    model = model.to(device)
    lightnet = lightnet.to(device)
    model.eval()
    lightnet.eval()

    weights = torch.log(torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0

    for img_file in tqdm(os.listdir(input_path)):
        # Read the image into a PIL Image 
        img = Image.open(os.path.join(input_path,img_file))
        interp = nn.Upsample(size=(img.size[1], img.size[0]), mode='bilinear', align_corners=True)
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        val_input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        image = val_input_transform(img)
        image = image.to(device)
        image = image.unsqueeze(0)
        with torch.no_grad():
            r = lightnet(image)
            enhancement = image + r
            if args.model == 'RefineNet':
                output2 = model(enhancement)
            else:
                _, output2 = model(enhancement)

        weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        output2 = output2 * weights_prob
        output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        name = img_file
        np.save('%s/%s.npy' % (output_path, name.split('.')[0]), (output<=10).transpose())
        output[output<=10] = 0
        output[output>10] = 1

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        # output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
        # img.save('%s/%s.png' % (args.save, name.split('.')[0]))

if __name__ == '__main__':
    main()
