import os
import queue
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import click
import hydra
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from fish_speech.datasets.text import CODEBOOK_EOS_TOKEN_ID, CODEBOOK_PAD_TOKEN_ID
from fish_speech.text import clean_text, split_text
import librosa
import soundfile as sf
from lightning import LightningModule
from omegaconf import OmegaConf
from fish_speech.utils.file import AUDIO_EXTENSIONS
# current_dir = os.path.dirname(os.path.abspath(__file__))
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
'---------------------------------------------------------'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from fish_speech.models.text2semantic.llama import DualARTransformer, NaiveTransformer


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)
    codebooks = [
        sample(
            x.logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]
    x = x.hidden_states

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor([codebook_idx], device=x.device, dtype=torch.long)
        logits = model.forward_generate_fast(x, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        x = model.fast_embeddings(a)
        codebooks.append(a)

    return torch.stack(codebooks, dim=0)


def decode_one_token_naive(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    codebooks = [
        sample(
            x.token_logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample(
                x.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[i + 1] if previous_tokens is not None else None
                ),
                **sampling_kwargs,
            )[0]
        )

    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    eos_token_id: int = 2,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if (
            cur_token[0, 0, -1] == eos_token_id
            or cur_token[0, 0, -1] == im_end_id
            or (cur_token[0, 1:, -1] == CODEBOOK_EOS_TOKEN_ID).any()
        ):
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: NaiveTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int = 2,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1, max_seq_len=T_new, dtype=next(model.parameters()).dtype
        )

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((codebook_dim, T_new), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar
    )
    next_token = prefill_decode(
        model, prompt.view(1, codebook_dim, -1), input_pos, **sampling_kwargs
    )
    seq[:, T : T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        eos_token_id=eos_token_id,
        im_end_id=im_end_id,
        decode_one_token=decode_one_token,
        **sampling_kwargs,
    )
    # x = torch.cat(generated_tokens, dim=1)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq


def encode_tokens(
    tokenizer,
    string,
    bos=True,
    device="cuda",
    prompt_tokens=None,
    speaker=None,
    num_codebooks=4,
):
    string = clean_text(string)

    if speaker is None:
        speaker = "assistant"

    string = (
        f"<|im_start|>user<|im_sep|>{string}<|im_end|><|im_start|>{speaker}<|im_sep|>"
    )
    if bos:
        string = f"<|begin_of_sequence|>{string}"

    new_tokens = tokenizer.encode(
        string,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )
    tokens = torch.tensor([new_tokens], dtype=torch.int, device=device)

    # Codebooks
    zeros = (
        torch.ones((num_codebooks, tokens.size(1)), dtype=torch.int, device=device)
        * CODEBOOK_PAD_TOKEN_ID
    )
    prompt = torch.cat((tokens, zeros), dim=0)

    if prompt_tokens is None:
        return prompt

    # Get prompt tokens
    if prompt_tokens.ndim == 3:
        assert (
            prompt_tokens.shape[0] == 1
        ), f"3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
        prompt_tokens = prompt_tokens[0]

    assert prompt_tokens.ndim == 2
    data = prompt_tokens + 2

    if prompt_tokens.shape[0] > num_codebooks:
        logger.warning(
            f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
        )
        data = data[:num_codebooks]

    # Add eos token for each codebook
    data = torch.cat(
        (
            data,
            torch.ones((data.size(0), 1), dtype=torch.int, device=device)
            * CODEBOOK_EOS_TOKEN_ID,
        ),
        dim=1,
    )

    # Since 1.0, we use <|semantic|>
    s0_token_id = tokenizer.convert_tokens_to_ids("<|semantic|>")
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    main_token_ids = (
        torch.ones((1, data.size(1)), dtype=torch.int, device=device) * s0_token_id
    )
    main_token_ids[0, -1] = end_token_id

    data = torch.cat((main_token_ids, data), dim=0)
    prompt = torch.cat((prompt, data), dim=1)

    return prompt


def load_model1(
    config_name, checkpoint_path, device, precision, max_length, compile=False
):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="fish_speech/configs/model"):
        cfg = compose(
            config_name=config_name, overrides=[f"config.max_seq_len={max_length}"]
        )

    model: Union[NaiveTransformer, DualARTransformer] = instantiate(cfg)

    if "int8" in str(checkpoint_path):
        logger.info("Using int8 weight-only quantization!")
        from .quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        logger.info("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from .quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if any(k.startswith("model.") for k in checkpoint):
        checkpoint = {
            k.replace("model.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("model.")
        }

    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    logger.info("Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        decode_one_token = decode_one_token_naive
        logger.info("Using NaiveTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    tokenizer: callable,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 2048,
    chunk_length: int = 150,
    speaker: Optional[str] = None,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    encoded_prompts = []

    if use_prompt:
        for idx, (t, c) in enumerate(zip(prompt_text, prompt_tokens)):
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=t,
                    bos=idx == 0,
                    device=device,
                    prompt_tokens=c,
                    speaker=speaker,
                    num_codebooks=model.config.num_codebooks,
                )
            )

    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                bos=idx == 0 and not use_prompt,
                device=device,
                speaker=speaker,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0

        while seg_idx < len(encoded):
            logger.info(
                f"Generating sentence {seg_idx + 1}/{len(encoded)} of sample {sample_idx + 1}/{num_samples}"
            )

            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last 2000 tokens
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 1024 - sum(
                    t.shape[1] for t in encoded_prompts
                ):
                    break
            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list, always make sure first segment is included to avoid drift
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[:2] + global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            if use_prompt:
                partial_encoded = encoded_prompts + partial_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                im_end_id=im_end_id,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if sample_idx == 0 and seg_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            if torch.cuda.is_available():
                logger.info(
                    f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
                )

            # Put the generated tokens
            # since there is <im_end> and <eos> tokens, we remove last 2 tokens
            codes = y[1:, prompt_length:-2].clone()

            codes = codes - 2
            assert (codes >= 0).all(), f"Negative code found"

            decoded = y[:, prompt_length:-1].clone()
            if decoded[0, -1] != im_end_id:  # <im_end>
                val = [[im_end_id]] + [[CODEBOOK_EOS_TOKEN_ID]] * (decoded.size(0) - 1)
                decoded = torch.cat(
                    (decoded, torch.tensor(val, device=device, dtype=torch.int)), dim=1
                )

            # But for global encoding, we should keep the <im_end> token
            global_encoded.append(decoded)
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            yield GenerateResponse(action="sample", codes=codes, text=texts[seg_idx])
            seg_idx += 1

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    config_name,
    checkpoint_path,
    device,
    precision,
    max_length: int,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = load_model1(
            config_name, checkpoint_path, device, precision, max_length, compile=compile
        )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


num_samples = 2
max_new_tokens = 0
top_p = 0.7
repetition_penalty = 1.5
temperature = 0.7
checkpoint_path = os.path.join(current_dir, "checkpoints/text2semantic-sft-medium-v1.1-4k.pth")
config_name = "dual_ar_2_codebook_medium"
compile = True
seed = 42
speaker = None
half = False
iterative_prompt = True
max_length = 2048
chunk_length = 150
device = "cuda"

precision = torch.half if half else torch.bfloat16
logger.info("Loading model ...")
t0 = time.time()
model1, decode_one_token = load_model1(
    config_name, checkpoint_path, device, precision, max_length, compile=compile
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")



'=================================================================='
# register eval resolver
OmegaConf.register_new_resolver("eval", eval)
def load_model(config_name, checkpoint_path, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="fish_speech/configs"):
        cfg = compose(config_name=config_name)
    model: LightningModule = instantiate(cfg.model)
    state_dict = torch.load(
        checkpoint_path,
        map_location=model.device,
    )

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    logger.info("Restored model from checkpoint")

    return model

tokenizer_name = "fishaudio/fish-speech-1"
output_path = os.path.join(current_dir, "fake.wav")
config_name = "vits_decoder_finetune"
# checkpoint_path = "//home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/fish-speech/fishspeech/checkpoints/vits_decoder_v1.1.ckpt"
checkpoint_path = "./checkpoints/vits_decoder_v1.1.ckpt"
device='cuda'
model = load_model(config_name, checkpoint_path, device=device)
def npygen(text: str, prompt_text: Optional[list[str]], prompt_tokens: Optional[list[Path]]):
    global tokenizer_name
    if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    generator = generate_long(
        model=model1,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        tokenizer=tokenizer,
        compile=compile,
        speaker=speaker,
        iterative_prompt=iterative_prompt,
        max_length=max_length,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                np.save(f"codes_{idx}.npy", torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to codes_{idx}.npy")
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


def vits_inference(input_path, reference_path, text):
    assert input_path.suffix == ".npy", f"Expected .npy file, got {input_path.suffix}"
    global tokenizer_name
    logger.info(f"Processing precomputed indices from {input_path}")
    indices = np.load(input_path)
    indices = torch.from_numpy(indices).to(model.device).long()
    assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"

    # Extract reference audio
    if reference_path is not None:
        assert (
            reference_path.suffix in AUDIO_EXTENSIONS
        ), f"Expected audio file, got {reference_path.suffix}"
        reference_audio, sr = librosa.load(reference_path, sr=model.sampling_rate)
        reference_audio = torch.from_numpy(reference_audio).to(model.device).float()
        reference_spec = model.spec_transform(reference_audio[None])
        reference_embedding = model.generator.encode_ref(
            reference_spec,
            torch.tensor([reference_spec.shape[-1]], device=model.device),
        )
        logger.info(
            f"Loaded reference audio from {reference_path}, shape: {reference_audio.shape}"
        )
    else:
        reference_embedding = torch.zeros(
            1, model.generator.gin_channels, 1, device=model.device
        )
        logger.info("No reference audio provided, use zero embedding")

    # Extract text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded_text = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    logger.info(f"Encoded text: {encoded_text.shape}")

    # Restore
    feature_lengths = torch.tensor([indices.shape[1]], device=model.device)
    quantized = model.generator.vq.indicies_to_vq_features(
        indices=indices[None], feature_lengths=feature_lengths
    )
    logger.info(f"Restored VQ features: {quantized.shape}")

    # Decode
    fake_audios = model.generator.decode(
        quantized,
        torch.tensor([quantized.shape[-1]], device=model.device),
        encoded_text,
        torch.tensor([encoded_text.shape[-1]], device=model.device),
        ge=reference_embedding,
    )
    logger.info(
        f"Generated audio: {fake_audios.shape}, equivalent to {fake_audios.shape[-1] / model.sampling_rate:.2f} seconds"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.sampling_rate)
    logger.info(f"Saved audio to {output_path}")

file = './rolewav/roleinfo.json'
import json
with open(file, 'r', encoding='utf-8') as f:
    dt = json.load(f)


yuyin_file = './rolewav/roleinfo.json'
with open(yuyin_file, 'r', encoding='utf-8') as f:
    yuyin_data = json.load(f)


# role: ['Klee', 'Qiqi', "ZhongLi"]

def get_audio(role, text):
    npygen(text, prompt_text=[dt[role]['text']], prompt_tokens=[dt[role]['file'] + ".npy"])
    vits_inference(input_path=Path("codes_0.npy"), reference_path=Path(dt[role]['file'] + ".wav"), text=text)