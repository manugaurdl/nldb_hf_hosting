from typing import Tuple
import clip
import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LogitsProcessorList,
    LogitsProcessor,
)
from huggingface_hub import PyTorchModelHubMixin

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.stop_word_ids = set(
            [
                idx
                for idx in range(len(tokenizer))
                if "." in tokenizer.convert_ids_to_tokens(idx)
            ]
        )
        self.vocab_size = len(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for i, input_id in enumerate(input_ids):
            if input_id[-1].item() in self.stop_word_ids:
                scores[i, : self.vocab_size] = torch.finfo().min
                scores[i, self.vocab_size :] = float("-inf")
                scores[i, self.eos_token_id] = 0.0
        return scores


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCapModel(nn.Module):
    def __init__(
        self,
        clip_prefix_size: int,
        nb_prefix_tokens: int = 10,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapModel, self).__init__()

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.do_sample = do_sample
        self.beam_size = beam_size
        self.max_len = max_len

        self.logits_processor = StopTokenLogitsProcessor(self.tokenizer)

        gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        input_dim = clip_prefix_size

        self.nb_prefix_tokens = nb_prefix_tokens

        hidden_dim = (gpt_embedding_size * self.nb_prefix_tokens) // 2
        output_dim = gpt_embedding_size * self.nb_prefix_tokens
        self.clip_project = MLP((input_dim, hidden_dim, output_dim))

    def forward(
        self,
        image_feats,
        greedy_baseline=False,
    ):
        prompts = self.clip_project(image_feats)
        prompts = prompts.view(image_feats.shape[0], self.nb_prefix_tokens, -1)

        bsz, prefix_len, h_dim = prompts.shape

        prompts_flat = prompts.view(-1, prompts.size(-1))
        start = len(self.tokenizer)
        end = start + (bsz * prefix_len)
        input_ids = torch.arange(start, end).view(*prompts.shape[:2]).to(prompts.device)
        self.gpt.get_input_embeddings().weight.data[start:end] = prompts_flat

        if not greedy_baseline and self.training:
            self.do_sample = True
            temp = 0.3
        else:
            temp = 1.0

        generated = self.gpt.generate(
            input_ids,
            do_sample=self.do_sample,
            max_length=self.max_len,
            num_beams=self.beam_size,
            num_return_sequences=1,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            top_k=len(self.tokenizer),
        )

        indices = generated[:, prefix_len:]

        suffix = self.gpt.get_input_embeddings()(indices)
        inputs_embeds = torch.cat([prompts, suffix], dim=1)

        logits = self.gpt(inputs_embeds=inputs_embeds)
        logits = logits[0][:, prefix_len - 1 : -1, : len(self.tokenizer)]
        logits = logits / (temp if temp > 0 else 1.0)
        logits = logits.log_softmax(-1)

        max_k = indices.size(1)
        end_of_caption = indices == self.eos_token_id
        extra_tokens = end_of_caption.cumsum(dim=1) > 0
        msg_lengths = max_k - (extra_tokens).sum(dim=1)
        mask = (extra_tokens == 0).float()

        log_probs = torch.gather(logits, dim=2, index=indices.unsqueeze(2)).squeeze(-1)
        log_probs *= mask
        log_probs = log_probs.sum(1) / msg_lengths

        decoded_captions = self.tokenizer.batch_decode(
            indices,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        self.do_sample = False

        return decoded_captions, log_probs, torch.randn(1)

    def maybe_patch_gpt(self, max_embeddings):
        if not getattr(self.gpt, "_patched", False):
            self.gpt._patched = True
            self.gpt.resize_token_embeddings(len(self.tokenizer) + max_embeddings)

            if self.gpt.get_output_embeddings().bias is None:
                self.gpt.get_output_embeddings().bias = torch.nn.Parameter(
                    torch.tensor([0.0] * (len(self.tokenizer) + max_embeddings))
                )
                self.gpt.get_output_embeddings().bias.requires_grad = False
                self.gpt.get_output_embeddings().to(
                    self.gpt.get_output_embeddings().weight.device
                )
                self.gpt._originally_with_no_bias = True
            else:
                self.gpt._originally_with_no_bias = False
            self.gpt.get_output_embeddings().bias.data[-max_embeddings:] = float("-inf")

    def maybe_unpatch_gpt(self):
        if getattr(self.gpt, "_patched", False):
            self.gpt._patched = False
            self.gpt.resize_token_embeddings(len(self.tokenizer))
            if self.gpt._originally_with_no_bias:
                self.gpt.get_output_embeddings().bias = None


class ClipCapSender(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        clip_model: str,
        clipcap_path: str,
        official_clipcap_weights: str,
        best_model_path: str,
        train_method: str,
        do_sample: bool = False,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapSender, self).__init__()

        self.clip, self.clip_preproc = clip.load(clip_model)
        convert_models_to_fp32(self.clip)

        self.clipcap = ClipCapModel(
            clip_prefix_size=self.clip.visual.output_dim,
            do_sample=do_sample,
            beam_size=beam_size,
            max_len=max_len,
        )
        if train_method != "mle":
            if clipcap_path is not None:
                print(f"| LOADED MODEL : {clipcap_path}")
                desired_format_state_dict = torch.load(official_clipcap_weights)
                saved_state_dict = torch.load(clipcap_path)

                state_dict = {}
                for idx, k in enumerate(desired_format_state_dict.keys()):
                    state_dict[k] = saved_state_dict["sender.clipcap." + k]

                self.clipcap.load_state_dict(state_dict)
        
        # Load best model
        self.load_best_model(best_model_path)

    def load_best_model(self, best_model_path):
        self.unpatch_model()

        trained_wts = torch.load(best_model_path)
        updated_wts = self.state_dict().copy()

        for k in list(self.state_dict().keys()):
            if "sender." + k in trained_wts:
                updated_wts[k] = trained_wts["sender." + k]
            elif k in trained_wts:
                updated_wts[k] = trained_wts[k]

        self.load_state_dict(updated_wts)
        self.patch_model(batch_size=32, prefix_len=10)

    def forward(self, images: torch.Tensor, greedy_baseline=False):
        image_feats = self.clip.visual(images)
        captions, log_probs, kl_div = self.clipcap(image_feats, greedy_baseline)
        return captions, log_probs, kl_div

    def train(self, mode: bool = True):
        self.training = mode
        self.clipcap.train(mode)
        return self

    def patch_model(self, batch_size: int = 500, prefix_len: int = 10):
        self.clipcap.maybe_patch_gpt(batch_size * prefix_len)

    def unpatch_model(self):
        self.clipcap.maybe_unpatch_gpt()