from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import (
    FixedWindowCharDataset,
    generate_text,
    render_predictions,
    resolve_device,
    set_seed,
    train_fixed_batch,
    train_tiny_dataset,
)
from core.tiny_char_transformer import (
    PlainResidualCombine,
    TinyTransformerCharModel,
    count_parameters,
)
from experiments.pytorch_char_predictive_chain import (
    PredictiveChainCharModel,
    compute_losses,
    train_on_fixed_batch,
    train_on_tiny_dataset,
)
from experiments.pytorch_char_rnn_baseline import TinyRnnCharModel
from experiments.pytorch_char_sanity_check import FeedForwardCharModel


SHAKESPEARE_SNIPPET = """HAMLET:
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry,
And lose the name of action.

VIOLA:
Make me a willow cabin at your gate,
And call upon my soul within the house.

HAMLET:
What a piece of work is a man! how noble in reason! how infinite in faculty!
in form and moving how express and admirable! in action how like an angel!
in apprehension how like a god! the beauty of the world! the paragon of animals!
And yet, to me, what is this quintessence of dust? man delights not me: no, nor woman neither.

MACBETH:
Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?
I see thee yet, in form as palpable
As this which now I draw.
Thou marshall'st me the way that I was going;
And such an instrument I was to use.
Mine eyes are made the fools o' the other senses,
Or else worth all the rest; I see thee still,
And on thy blade and dudgeon gouts of blood,
Which was not so before. There's no such thing:
It is the bloody business which informs
Thus to mine eyes.

MACBETH:
Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
That struts and frets his hour upon the stage,
And then is heard no more: it is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

MARK ANTONY:
Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest--
For Brutus is an honourable man;
So are they all, all honourable men--
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.
He hath brought many captives home to Rome
Whose ransoms did the general coffers fill:
Did this in Caesar seem ambitious?
When that the poor have cried, Caesar hath wept:
Ambition should be made of sterner stuff:
Yet Brutus says he was ambitious;
And Brutus is an honourable man.

PORTIA:
The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath: it is twice blest;
It blesseth him that gives and him that takes:
'Tis mightiest in the mightiest: it becomes
The throned monarch better than his crown;
His sceptre shows the force of temporal power,
The attribute to awe and majesty,
Wherein doth sit the dread and fear of kings;
But mercy is above this sceptred sway;
It is enthroned in the hearts of kings,
It is an attribute to God himself;
And earthly power doth then show likest God's
When mercy seasons justice.

JAQUES:
All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms.
And then the whining school-boy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part.

KING HENRY:
Once more unto the breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a man
As modest stillness and humility:
But when the blast of war blows in our ears,
Then imitate the action of the tiger;
Stiffen the sinews, summon up the blood,
Disguise fair nature with hard-favour'd rage;
Then lend the eye a terrible aspect;
Now set the teeth and stretch the nostril wide,
Hold hard the breath and bend up every spirit
To his full height. On, on, you noblest English,
Whose blood is fet from fathers of war-proof!

LEAR:
Blow, winds, and crack your cheeks! rage! blow!
You cataracts and hurricanoes, spout
Till you have drench'd our steeples, drown'd the cocks!
You sulphurous and thought-executing fires,
Vaunt-couriers to oak-cleaving thunderbolts,
Singe my white head! And thou, all-shaking thunder,
Smite flat the thick rotundity o' the world!
Crack nature's moulds, all germens spill at once,
That make ingrateful man!

PROSPERO:
Our revels now are ended. These our actors,
As I foretold you, were all spirits and
Are melted into air, into thin air:
And, like the baseless fabric of this vision,
The cloud-capp'd towers, the gorgeous palaces,
The solemn temples, the great globe itself,
Yea, all which it inherit, shall dissolve,
And, like this insubstantial pageant faded,
Leave not a rack behind. We are such stuff
As dreams are made on, and our little life
Is rounded with a sleep.

VIOLA:
She never told her love,
But let concealment, like a worm i' the bud,
Feed on her damask cheek: she pined in thought,
And with a green and yellow melancholy
She sat like patience on a monument,
Smiling at grief. Was not this love indeed?
We men may say more, swear more: but indeed
Our shows are more than will; for still we prove
Much in our vows, but little in our love.
"""


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 5
    train_fraction: float = 0.8
    embedding_dim: int = 24
    feedforward_hidden_dim: int = 64
    rnn_hidden_dim: int = 64
    transformer_d_model: int = 24
    transformer_feedforward_dim: int = 112
    transformer_num_heads: int = 4
    transformer_num_layers: int = 1
    predictive_num_nodes: int = 8
    predictive_hidden_dim: int = 10
    predictive_message_dim: int = 24
    predictive_detach_messages: bool = True
    overfit_batch_size: int = 32
    overfit_steps: int = 3000
    overfit_learning_rate: float = 0.02
    train_batch_size: int = 128
    train_steps: int = 1200
    train_learning_rate: float = 0.01
    predictive_gradient_clip_norm: float = 1.0
    sample_length: int = 120
    seed: int = 7


@dataclass(frozen=True)
class DatasetSplit:
    train_inputs: Tensor
    train_targets: Tensor
    val_inputs: Tensor
    val_targets: Tensor
    split_index: int
    train_text: str
    val_text: str


def format_float_token(value: float) -> str:
    return format(value, "g").replace("-", "neg").replace(".", "p")


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def encode_windows(
    text: str,
    *,
    context_size: int,
    stoi: dict[str, int],
) -> tuple[Tensor, Tensor]:
    if len(text) <= context_size:
        raise ValueError("Text split must be longer than the context size.")

    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    inputs = []
    targets = []
    for start in range(len(encoded) - context_size):
        stop = start + context_size
        inputs.append(encoded[start:stop])
        targets.append(encoded[stop])
    return torch.stack(inputs), torch.stack(targets)


def build_train_val_split(
    text: str,
    *,
    context_size: int,
    stoi: dict[str, int],
    train_fraction: float,
) -> DatasetSplit:
    split_index = int(len(text) * train_fraction)
    train_text = text[:split_index]
    val_text = text[split_index:]
    if len(train_text) <= context_size or len(val_text) <= context_size:
        raise ValueError("Train/validation split leaves too little text for windows.")

    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(
            f"Validation text contains characters absent from training text: {missing_val_chars}"
        )

    train_inputs, train_targets = encode_windows(
        train_text,
        context_size=context_size,
        stoi=stoi,
    )
    val_inputs, val_targets = encode_windows(
        val_text,
        context_size=context_size,
        stoi=stoi,
    )
    return DatasetSplit(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        split_index=split_index,
        train_text=train_text,
        val_text=val_text,
    )


def select_prompts(train_text: str, val_text: str, *, context_size: int) -> list[str]:
    prompts = [
        train_text[:context_size],
        train_text[len(train_text) // 2 : len(train_text) // 2 + context_size],
        val_text[:context_size],
    ]
    deduped = []
    for prompt in prompts:
        if len(prompt) == context_size and prompt not in deduped:
            deduped.append(prompt)
    if len(deduped) < 3:
        for start in range(0, len(train_text) - context_size + 1, context_size):
            prompt = train_text[start : start + context_size]
            if prompt not in deduped:
                deduped.append(prompt)
            if len(deduped) == 3:
                break
    return deduped


def evaluate_standard_model(model: nn.Module, inputs: Tensor, targets: Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(inputs)
        task_loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return {
        "train_or_eval_loss": task_loss.item(),
        "task_loss": task_loss.item(),
        "accuracy": accuracy,
    }


def evaluate_predictive_chain_model(
    model: PredictiveChainCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    auxiliary_weight: float,
) -> dict[str, float]:
    with torch.no_grad():
        losses = compute_losses(
            model,
            inputs,
            targets,
            auxiliary_weight=auxiliary_weight,
        )
    return {
        "train_or_eval_loss": losses.task_loss.item(),
        "task_loss": losses.task_loss.item(),
        "total_loss": losses.total_loss.item(),
        "auxiliary_total": losses.auxiliary_total.item(),
        "accuracy": losses.accuracy,
    }


def save_generation_samples(
    path: Path,
    *,
    model: nn.Module,
    dataset: FixedWindowCharDataset,
    prompts: list[str],
    sample_length: int,
    device: torch.device,
) -> dict[str, str]:
    with torch.no_grad():
        samples = {
            prompt.replace("\n", "\\n"): generate_text(
                model,
                dataset,
                prompt,
                length=sample_length,
                device=device,
            )
            for prompt in prompts
        }
    write_json(path, samples)
    return samples


def save_validation_predictions(
    path: Path,
    *,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    predictions: Tensor,
) -> None:
    path.write_text(
        render_predictions(
            dataset,
            inputs.cpu(),
            targets.cpu(),
            predictions.cpu(),
        ),
        encoding="utf-8",
    )


def save_standard_model_run(
    *,
    model_name: str,
    output_dir: Path,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    build_model: callable,
    parameter_count: int,
    overfit_batch_size: int,
    overfit_steps: int,
    overfit_learning_rate: float,
    train_batch_size: int,
    train_steps: int,
    train_learning_rate: float,
    prompts: list[str],
    sample_length: int,
    device: torch.device,
) -> dict[str, object]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    overfit_model = build_model().to(device)
    overfit_inputs = train_inputs[:overfit_batch_size]
    overfit_targets = train_targets[:overfit_batch_size]
    overfit_trace, overfit_loss, overfit_accuracy = train_fixed_batch(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        steps=overfit_steps,
        learning_rate=overfit_learning_rate,
    )
    overfit_predictions = overfit_model(overfit_inputs).argmax(dim=1)
    overfit_reached = overfit_accuracy == 1.0 and overfit_loss < 1e-3
    write_json(
        model_dir / "overfit_metrics.json",
        {
            "final_loss": overfit_loss,
            "final_accuracy": overfit_accuracy,
            "reached_memorization_bar": overfit_reached,
            "trace": overfit_trace,
        },
    )
    save_validation_predictions(
        model_dir / "overfit_predictions.txt",
        dataset=dataset,
        inputs=overfit_inputs,
        targets=overfit_targets,
        predictions=overfit_predictions,
    )
    if not overfit_reached:
        raise RuntimeError(f"{model_name} failed the one-batch overfit check.")

    train_model = build_model().to(device)
    started_at = time.perf_counter()
    train_trace, _train_loss, _train_accuracy = train_tiny_dataset(
        train_model,
        train_inputs,
        train_targets,
        batch_size=train_batch_size,
        steps=train_steps,
        learning_rate=train_learning_rate,
    )
    runtime_seconds = time.perf_counter() - started_at
    train_metrics = evaluate_standard_model(train_model, train_inputs, train_targets)
    val_metrics = evaluate_standard_model(train_model, val_inputs, val_targets)
    with torch.no_grad():
        val_predictions = train_model(val_inputs[:32]).argmax(dim=1)
    save_validation_predictions(
        model_dir / "validation_predictions.txt",
        dataset=dataset,
        inputs=val_inputs[:32],
        targets=val_targets[:32],
        predictions=val_predictions,
    )
    samples = save_generation_samples(
        model_dir / "samples.json",
        model=train_model,
        dataset=dataset,
        prompts=prompts,
        sample_length=sample_length,
        device=device,
    )
    write_json(
        model_dir / "train_val_metrics.json",
        {
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_trace": train_trace,
        },
    )
    return {
        "model_name": model_name,
        "parameter_count": parameter_count,
        "overfit_reached": overfit_reached,
        "runtime_seconds": runtime_seconds,
        "train_loss": train_metrics["task_loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["task_loss"],
        "val_accuracy": val_metrics["accuracy"],
        "samples": samples,
    }


def save_predictive_chain_run(
    *,
    model_name: str,
    output_dir: Path,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    message_dim: int,
    num_nodes: int,
    detach_messages: bool,
    auxiliary_weight: float,
    overfit_batch_size: int,
    overfit_steps: int,
    overfit_learning_rate: float,
    train_batch_size: int,
    train_steps: int,
    train_learning_rate: float,
    gradient_clip_norm: float,
    prompts: list[str],
    sample_length: int,
    device: torch.device,
) -> dict[str, object]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    def build_model() -> PredictiveChainCharModel:
        return PredictiveChainCharModel(
            vocab_size=vocab_size,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            detach_messages=detach_messages,
        )

    parameter_count = count_parameters(build_model())
    write_json(
        model_dir / "model_summary.json",
        {
            "parameter_count": parameter_count,
            "num_nodes": num_nodes,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "message_dim": message_dim,
            "detach_messages": detach_messages,
            "auxiliary_weight": auxiliary_weight,
        },
    )

    overfit_model = build_model().to(device)
    overfit_inputs = train_inputs[:overfit_batch_size]
    overfit_targets = train_targets[:overfit_batch_size]
    overfit_trace, overfit_losses = train_on_fixed_batch(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        steps=overfit_steps,
        learning_rate=overfit_learning_rate,
        auxiliary_weight=auxiliary_weight,
        gradient_clip_norm=gradient_clip_norm,
    )
    overfit_predictions = overfit_model(overfit_inputs).argmax(dim=1)
    overfit_reached = (
        overfit_losses.accuracy == 1.0 and overfit_losses.task_loss.item() < 1e-3
    )
    write_json(
        model_dir / "overfit_metrics.json",
        {
            "final_total_loss": overfit_losses.total_loss.item(),
            "final_task_loss": overfit_losses.task_loss.item(),
            "final_auxiliary_total": overfit_losses.auxiliary_total.item(),
            "final_accuracy": overfit_losses.accuracy,
            "reached_memorization_bar": overfit_reached,
            "trace": overfit_trace,
        },
    )
    save_validation_predictions(
        model_dir / "overfit_predictions.txt",
        dataset=dataset,
        inputs=overfit_inputs,
        targets=overfit_targets,
        predictions=overfit_predictions,
    )
    if not overfit_reached:
        raise RuntimeError(f"{model_name} failed the one-batch overfit check.")

    train_model = build_model().to(device)
    started_at = time.perf_counter()
    train_trace, _train_losses = train_on_tiny_dataset(
        train_model,
        train_inputs,
        train_targets,
        batch_size=train_batch_size,
        steps=train_steps,
        learning_rate=train_learning_rate,
        auxiliary_weight=auxiliary_weight,
        gradient_clip_norm=gradient_clip_norm,
    )
    runtime_seconds = time.perf_counter() - started_at
    train_metrics = evaluate_predictive_chain_model(
        train_model,
        train_inputs,
        train_targets,
        auxiliary_weight=auxiliary_weight,
    )
    val_metrics = evaluate_predictive_chain_model(
        train_model,
        val_inputs,
        val_targets,
        auxiliary_weight=auxiliary_weight,
    )
    with torch.no_grad():
        val_predictions = train_model(val_inputs[:32]).argmax(dim=1)
    save_validation_predictions(
        model_dir / "validation_predictions.txt",
        dataset=dataset,
        inputs=val_inputs[:32],
        targets=val_targets[:32],
        predictions=val_predictions,
    )
    samples = save_generation_samples(
        model_dir / "samples.json",
        model=train_model,
        dataset=dataset,
        prompts=prompts,
        sample_length=sample_length,
        device=device,
    )
    write_json(
        model_dir / "train_val_metrics.json",
        {
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_trace": train_trace,
        },
    )
    return {
        "model_name": model_name,
        "parameter_count": parameter_count,
        "overfit_reached": overfit_reached,
        "runtime_seconds": runtime_seconds,
        "train_loss": train_metrics["task_loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["task_loss"],
        "val_accuracy": val_metrics["accuracy"],
        "train_total_loss": train_metrics["total_loss"],
        "val_total_loss": val_metrics["total_loss"],
        "train_auxiliary_total": train_metrics["auxiliary_total"],
        "val_auxiliary_total": val_metrics["auxiliary_total"],
        "samples": samples,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = (
        repo_root
        / "research"
        / "questions"
        / "predictive-chain"
        / "artifacts"
        / "shakespeare_comparison"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--predictive-message-dim", type=int)
    parser.add_argument("--predictive-aux-weights", type=float, nargs="+", default=[1.0, 0.001])
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    config = RunConfig()
    if args.predictive_message_dim is not None:
        config = replace(config, predictive_message_dim=args.predictive_message_dim)
    set_seed(config.seed)
    device = resolve_device(args.device)

    corpus_text = SHAKESPEARE_SNIPPET
    dataset = FixedWindowCharDataset(corpus_text, context_size=config.context_size)
    split = build_train_val_split(
        corpus_text,
        context_size=config.context_size,
        stoi=dataset.stoi,
        train_fraction=config.train_fraction,
    )
    prompts = select_prompts(
        split.train_text,
        split.val_text,
        context_size=config.context_size,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "predictive_aux_weights": args.predictive_aux_weights,
            "skip_baselines": args.skip_baselines,
        },
    )
    git_status_short = current_git_status_short()
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
        },
    )
    write_json(
        output_dir / "corpus_summary.json",
        {
            "corpus_name": "hardcoded_shakespeare_excerpt",
            "corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
            "total_characters": len(corpus_text),
            "vocab_size": dataset.vocab_size,
            "split_index": split.split_index,
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "prompts": [prompt.replace("\n", "\\n") for prompt in prompts],
            "train_text_preview": split.train_text[:250].replace("\n", "\\n"),
            "val_text_preview": split.val_text[:250].replace("\n", "\\n"),
        },
    )

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    results = []
    for auxiliary_weight in args.predictive_aux_weights:
        results.append(
            save_predictive_chain_run(
                model_name=f"predictive_chain_aux_{format_float_token(auxiliary_weight)}_detach",
                output_dir=output_dir,
                dataset=dataset,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                vocab_size=dataset.vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.predictive_hidden_dim,
                message_dim=config.predictive_message_dim,
                num_nodes=config.predictive_num_nodes,
                detach_messages=config.predictive_detach_messages,
                auxiliary_weight=auxiliary_weight,
                overfit_batch_size=config.overfit_batch_size,
                overfit_steps=config.overfit_steps,
                overfit_learning_rate=config.overfit_learning_rate,
                train_batch_size=config.train_batch_size,
                train_steps=config.train_steps,
                train_learning_rate=config.train_learning_rate,
                gradient_clip_norm=config.predictive_gradient_clip_norm,
                prompts=prompts,
                sample_length=config.sample_length,
                device=device,
            )
        )

    if not args.skip_baselines:
        transformer_parameter_count = count_parameters(
            TinyTransformerCharModel(
                vocab_size=dataset.vocab_size,
                context_size=config.context_size,
                d_model=config.transformer_d_model,
                num_heads=config.transformer_num_heads,
                num_layers=config.transformer_num_layers,
                feedforward_dim=config.transformer_feedforward_dim,
                residual_factory=PlainResidualCombine,
            )
        )
        results.append(
            save_standard_model_run(
                model_name="transformer_baseline",
                output_dir=output_dir,
                dataset=dataset,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                build_model=lambda: TinyTransformerCharModel(
                    vocab_size=dataset.vocab_size,
                    context_size=config.context_size,
                    d_model=config.transformer_d_model,
                    num_heads=config.transformer_num_heads,
                    num_layers=config.transformer_num_layers,
                    feedforward_dim=config.transformer_feedforward_dim,
                    residual_factory=PlainResidualCombine,
                ),
                parameter_count=transformer_parameter_count,
                overfit_batch_size=config.overfit_batch_size,
                overfit_steps=config.overfit_steps,
                overfit_learning_rate=config.overfit_learning_rate,
                train_batch_size=config.train_batch_size,
                train_steps=config.train_steps,
                train_learning_rate=config.train_learning_rate,
                prompts=prompts,
                sample_length=config.sample_length,
                device=device,
            )
        )

        rnn_parameter_count = count_parameters(
            TinyRnnCharModel(
                vocab_size=dataset.vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.rnn_hidden_dim,
                num_layers=1,
                nonlinearity="tanh",
            )
        )
        results.append(
            save_standard_model_run(
                model_name="rnn_baseline",
                output_dir=output_dir,
                dataset=dataset,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                build_model=lambda: TinyRnnCharModel(
                    vocab_size=dataset.vocab_size,
                    embedding_dim=config.embedding_dim,
                    hidden_dim=config.rnn_hidden_dim,
                    num_layers=1,
                    nonlinearity="tanh",
                ),
                parameter_count=rnn_parameter_count,
                overfit_batch_size=config.overfit_batch_size,
                overfit_steps=config.overfit_steps,
                overfit_learning_rate=config.overfit_learning_rate,
                train_batch_size=config.train_batch_size,
                train_steps=config.train_steps,
                train_learning_rate=config.train_learning_rate,
                prompts=prompts,
                sample_length=config.sample_length,
                device=device,
            )
        )

        feedforward_parameter_count = count_parameters(
            FeedForwardCharModel(
                vocab_size=dataset.vocab_size,
                context_size=config.context_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.feedforward_hidden_dim,
            )
        )
        results.append(
            save_standard_model_run(
                model_name="feedforward_baseline",
                output_dir=output_dir,
                dataset=dataset,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                build_model=lambda: FeedForwardCharModel(
                    vocab_size=dataset.vocab_size,
                    context_size=config.context_size,
                    embedding_dim=config.embedding_dim,
                    hidden_dim=config.feedforward_hidden_dim,
                ),
                parameter_count=feedforward_parameter_count,
                overfit_batch_size=config.overfit_batch_size,
                overfit_steps=config.overfit_steps,
                overfit_learning_rate=config.overfit_learning_rate,
                train_batch_size=config.train_batch_size,
                train_steps=config.train_steps,
                train_learning_rate=config.train_learning_rate,
                prompts=prompts,
                sample_length=config.sample_length,
                device=device,
            )
        )

    write_json(
        output_dir / "comparison_summary.json",
        {
            "models": sorted(results, key=lambda row: row["val_loss"]),
            "best_by_val_loss": min(results, key=lambda row: row["val_loss"]),
            "best_by_val_accuracy": max(results, key=lambda row: row["val_accuracy"]),
        },
    )


if __name__ == "__main__":
    main()
