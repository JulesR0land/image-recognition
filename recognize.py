#!/bin/python3

from argparse import ArgumentParser

from torch import load, FloatTensor, max
from torchvision import transforms
from PIL import Image

from src.recognition_network import Network


classes = ("dog", "cat")


def parse_args():
    parser = ArgumentParser(
        prog="Recognize",
        description="Print the output of a pytorch model on an image.",
    )
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument("-w", "--width", type=int, metavar="width", help="The width of all images")
    parser.add_argument("-he", "--height", type=int, metavar="height", help="The height of all images")

    return parser.parse_args()


def get_tensor_image(image_path, width, height):
    img = Image.open(image_path)

    if width and height:
        img = img.resize((width, height))
    elif width:
        img = img.resize((width, img.height))
    elif width:
        img = img.resize((img.width, height))

    to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor = to_tensor(img)

    tensor = tensor.unsqueeze(0)

    return tensor


def get_model(model_path, input_shape):
    model = Network(input_shape)
    model.load_state_dict(load(model_path))

    return model


def main():
    args = parse_args()

    input = get_tensor_image(args.image_path, args.width, args.height)
    model = get_model(args.model_path, (input.shape[1], input.shape[2], input.shape[3]))

    output = model(input)

    _, predicted = max(output, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print("It\' a", classes[predicted[0]])



if __name__ == "__main__":
    main()