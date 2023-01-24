#!/bin/python3

from argparse import ArgumentParser
from glob import glob
from PIL import Image


from torch import load
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from src.recognition_network import Network
from src.trainer import train


def parse_args():
    parser = ArgumentParser(
        prog="train_model",
        description="Take a pre existing model or create a new one and train it on the datas passed in arguments.",
    )

    parser.add_argument("-m", "--model", type=str, metavar="model_path", help="Use a pre-existing model for training")
    parser.add_argument("-d", "--dataset", type=str, required=True, nargs=2, action="append", metavar=("name", "path"), help="The name and path of an image folder")
    parser.add_argument("-w", "--width", type=int, metavar="width", help="The width of all images")
    parser.add_argument("-he", "--height", type=int, metavar="height", help="The height of all images")

    return parser.parse_args()


def get_images_datas(datasets_info):
    datasets_paths = {}
    images = []
    classes = []

    for info in datasets_info:
        classes.append(info[0])
        datasets_paths[info[0]] = glob(info[1] + "/*.jpg")

    return classes, datasets_paths


def get_shape(width, height, datasets):
    if width is not None and height is not None:
        return (3, height, width)

    input_shape = [0, 0]

    for values in datasets.values():
        for image_path in values:
            with Image.open(image_path) as image:
                size = image.size
            if width is None and size[0] > input_shape[0]:
                input_shape[0] = size[0]
            if height is None and size[1] > input_shape[1]:
                input_shape[1] = size[1]

    return (3, input_shape[1], input_shape[0])


def format_datas(input_shape, datasets_paths):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    train_set = []
    test_set = []

    for i, values in enumerate(datasets_paths.values()):
        test_size = int(len(values) / 100 * 5)
        train_size = len(values) - test_size
        for j in range(0, train_size):
            with Image.open(values[j]).convert("RGB") as image:
                image = image.resize((input_shape[1], input_shape[2]))
                train_set.append((to_tensor(image), i))
        for j in range(train_size, len(values)):
            with Image.open(values[j]).convert("RGB") as image:
                image = image.resize((input_shape[1], input_shape[2]))
                test_set.append((to_tensor(image), i))

    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=True)

    return train_loader, test_loader


def load_training_req(model_path, input_shape):
    model = Network(input_shape)

    if model_path:
        model.load_state_dict(load(model_path))

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = CrossEntropyLoss()

    return model, optimizer, loss_fn



def main():
    args = parse_args()

    print(args)

    print("Getting images paths")

    classes, datasets_paths = get_images_datas(args.dataset)

    print("End of getting images paths")
    print("Getting input shape")

    input_shape = get_shape(args.width, args.height, datasets_paths)

    print("Input shape is", input_shape)
    print("Formatting images")

    train_loader, test_loader = format_datas(input_shape, datasets_paths)

    print("End of formatting images")
    print("Loading/Creating model")

    model, optimizer, loss_fn = load_training_req(args.model, input_shape)

    print("End of loading/creating model")
    print("Training...")

    train(model, loss_fn, optimizer, 15, train_loader, test_loader, classes)




if __name__ == "__main__":
    main()