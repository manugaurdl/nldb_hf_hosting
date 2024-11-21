class MaybeDoubleTransform:
    """Return sender and receiver images for an image-based communication game in EGG."""

    def __init__(self, sender_image_size: int, recv_image_size: int = None):
        self.sender_transform = self._get_transform(sender_image_size)
        self.recv_transform = None

        if recv_image_size:
            self.recv_transform = self._get_transform(recv_image_size)

    def _get_transform(self, image_size: int):
        def _convert_image_to_rgb(image: Image.Image):
            return image.convert("RGB")
        t = [
            transforms.Resize(image_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
        return transforms.Compose(t)

    def __call__(self, x):
        sender_image = self.sender_transform(x)
        # recv_image = sender_image
        # if self.recv_transform:
        #     recv_image = self.recv_transform(x)

        # return [sender_image, recv_image]
        return sender_image


def get_transform(sender_image_size: int, recv_image_size: int = None):
    return MaybeDoubleTransform(sender_image_size, recv_image_size)

#######################################################################################################################################
class CocoDataset:
    def __init__(self, root, samples, mle_train, split, caps_per_img, captions_type, max_len_token, prefix_len, transform, debug, mllm):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        # file_path, captions, image_id = self.samples[idx]
        #GET FILE PATH

        image = Image.open(os.path.join(self.root, file_path)).convert("RGB")
        sender_input = self.transform(image)


        return sender_input
