import cv2
from Feature_extraction_and_Enhancement_v2 import image_feature_extract
import os
from torchvision import transforms
import torch
from model_v2 import VisionTransformer
from PIL import Image
from config import parser
from gradient_recored import Gradient
import numpy as np
from lesion_localization import lesion_localization

args = parser.parse_args()
class_indict = {'0': 'CNV', '1': 'DME', '2': 'drusen', '3': 'normal'}

data_transform = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def train_before_feature_extract_and_enhance(filepath, destination_path):
    pathdir = os.listdir(filepath)
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    for allDir in pathdir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        frame = cv2.imread(child)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = image_feature_extract(img, False, True)
        cv2.imwrite(destination_path+'/'+allDir, result)


def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, :, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    frame = cv2.imread('F:/OCT2017/test/CNV/CNV-6256161-1.jpeg')
    frame = cv2.resize(frame, dsize=(512, 512))
    result = image_feature_extract(frame, False, False)

    result = frame
    img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    img = data_transform(img)

    img = torch.unsqueeze(img, dim=0)
    model = VisionTransformer(image_size=(args.image_size, args.image_size),
            patch_size=(args.patch_size, args.patch_size),
            emb_dim=args.emb_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_classes=args.num_class,
            attn_dropout_rate=args.attn_dropout_rate,
            dropout_rate=args.dropout_rate).to(select_device())

    model_weight_path = "./LLCT.pth"
    state_dict = torch.load(model_weight_path, map_location=select_device())
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()
    target_layer = model.module.transformer.encoder_layers[-1].norm1

    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()

    cam = Gradient(model=model, target_layer=target_layer, reshape_transform=reshape_transform)
    visual = cam(input_tensor=img, target_category=predict_cla)
    visual = visual[0, :]

    rgb_img = np.float32(frame) / 255

    st = class_indict[str(predict_cla)] + ' : %.3f' % predict[predict_cla].numpy()
    local = lesion_localization(frame, visual, class_indict[str(predict_cla)], predict[predict_cla].numpy())

    cv2.putText(result, class_indict[str(predict_cla)] + ' : %.3f' % predict[predict_cla].numpy(),
               (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (150, 0, 180), 2)
    possibility_of_normal = predict.numpy()[3]
    possibility_of_abnormal = 1 - possibility_of_normal
    cv2.imshow('local', local)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
