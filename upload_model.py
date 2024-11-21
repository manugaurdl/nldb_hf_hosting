from model import ClipCapSender
from config import CKPT_ID

def upload_to_hub():
    sender = ClipCapSender(
        clip_model="ViT-B/32",
        clipcap_path="./modded_mle.pt",
        official_clipcap_weights="./official_clipcap.pt",
        best_model_path="./best.pt",
        train_method="discriminative",
        do_sample=False,
        beam_size=5,
        max_len=50,
    )

    sender.eval()
    sender.push_to_hub(CKPT_ID)

def main():
    upload_to_hub()


if __name__ == "__main__":
    main()