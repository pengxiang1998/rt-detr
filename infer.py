import torch
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

def inference(source,score):
    ##################
    classes = ['car','truck',"bus"]
    print(source)
    img_path = source
    #############
    im = Image.open(img_path).convert('RGB')
    im = im.resize((640, 640))
    im_data = ToTensor()(im)[None]
    print(im_data.shape)

    size = torch.tensor([[640, 640]])
    sess = ort.InferenceSession("model.onnx")
    import time
    start = time.time()
    output = sess.run(
        output_names=['labels', 'boxes', 'scores'],
        #output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    )
    end = time.time()
    fps = 1.0 / (end - start)
    print(fps)


    labels, boxes, scores = output

    draw = ImageDraw.Draw(im)
    thrh = float(score)

    for i in range(im_data.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        num=sum(scr > thrh)
        #print(lab)
        print(f'box:{box}')
        for l, b in zip(lab, box):
            draw.rectangle(list(b), outline='red',)
            print(l.item())

            draw.text((b[0], b[1] - 10), font_size=20,text=str(classes[l.item()]), fill='blue', )
    #############
    im.save('static/img/result.jpg')
    return num,int(fps)
    #############