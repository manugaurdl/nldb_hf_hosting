from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import requests
from model import ClipCapSender
from config import CKPT_ID

def get_transform(image_size: int):
    def _convert_image_to_rgb(image: Image.Image):
        return image.convert("RGB")

    t = [
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return transforms.Compose(t)


def run_inference(url):
    sender = ClipCapSender.from_pretrained(CKPT_ID).to("cuda")

    image = Image.open(requests.get(url, stream=True).raw)
    t = get_transform(224)
    image = t(image).to("cuda")
    images = image[None, ...]

    captions, log_probs, kl_div = sender(images)
    print(captions)


def main():
    # Example
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    run_inference(url)


if __name__ == "__main__":
    main()
