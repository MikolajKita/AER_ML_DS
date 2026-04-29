import argparse
import csv
import logging
import os
import random
from itertools import islice
from pathlib import Path

import numpy as np
from river import datasets
from river.datasets import synth
from river.drift.binary import EDDM, DDM

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

from aer import ActiveExpertRepository
from aer_multi_model import ActiveExpertRepositoryMultiModel, ModelFactory
from recurring_stream import recurring_stream_generator
from shadow_expert_model import ShadowExpertModel
from plot_utils import plot_sliding_accuracy


MODEL_TYPES = ["HT", "ARF", "SRP", "NaiveBayes"]
DEFAULT_REAL_STREAM_LENGTH = 15000
DEFAULT_REAL_OUTPUT_DIR = "real_data_results"
DEFAULT_RIVER_DATA_DIR = ".river_data"

CSV_HEADER = [
    "evaluated_model_type",
    "wrapper_model_name",
    "chunk_index",
    "stream_label",
    "active_model",
    "period_accuracy",
    "cumulative_accuracy",
]


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    ModelFactory.GLOBAL_SEED = seed


def default_concepts(seed):
    return [
        ("Agrawal_Function_0", synth.Agrawal(seed=seed, classification_function=0)),
        # ("Agrawal_Function_2", synth.Agrawal(seed=seed, classification_function=2)),
        ("Agrawal_Function_6", synth.Agrawal(seed=seed, classification_function=6)),
    ]


def default_drift_detector():
    return EDDM(warm_start=30, alpha=0.95, beta=0.8)


def dataset_stream(dataset, stream_label):
    for x, y in dataset:
        yield x, y, stream_label


def phishing_stream():
    yield from dataset_stream(datasets.Phishing(), "Real_Phishing")


def elec2_stream():
    yield from dataset_stream(datasets.Elec2(), "Real_Elec2")


def synthetic_sea_drift_stream(position=10000, width=1000, seed=42):
    dataset = synth.ConceptDriftStream(
        stream=synth.SEA(seed=seed, variant=0),
        drift_stream=synth.SEA(seed=seed + 1, variant=1),
        position=position,
        width=width,
        seed=seed,
    )
    yield from dataset_stream(dataset, "Synthetic_SEA_Known_Drift")


DATASETS = {
    "synthetic_recurring": {
        "stream_factory": None,
        "stream_length": 3_000,
        "chunk_size": 500,
        "track_events": "drifts",
        "output_dir": ".",
        "known_drift": 500,
        "description": "Synthetic recurring Agrawal stream used by the original test.py run.",
    },
    "phishing": {
        "stream_factory": phishing_stream,
        "stream_length": DEFAULT_REAL_STREAM_LENGTH,
        "chunk_size": 250,
        "track_events": "model_changes",
        "output_dir": DEFAULT_REAL_OUTPUT_DIR,
        "known_drift": None,
        "description": "Small real binary stream, no known drift labels.",
    },
    "elec2": {
        "stream_factory": elec2_stream,
        "stream_length": DEFAULT_REAL_STREAM_LENGTH,
        "chunk_size": 1000,
        "track_events": "model_changes",
        "output_dir": DEFAULT_REAL_OUTPUT_DIR,
        "known_drift": None,
        "description": "Long real electricity-pricing stream with natural concept drift.",
    },
    "synthetic_sea_drift": {
        "stream_factory": synthetic_sea_drift_stream,
        "stream_length": DEFAULT_REAL_STREAM_LENGTH,
        "chunk_size": 1000,
        "track_events": "model_changes",
        "output_dir": DEFAULT_REAL_OUTPUT_DIR,
        "known_drift": [10000],
        "description": "Synthetic non-recurrent stream with a known drift centered at 10,000.",
    },
}


def build_models(model_type, available_model_types, drift_detector_factory):
    return {
        f"AER_{model_type}": ActiveExpertRepository(
            base_estimator=ModelFactory.createModel(model_type),
            drift_detector=drift_detector_factory(),
        ),
        "AER_MultiModel": ActiveExpertRepositoryMultiModel(
            base_estimator=ModelFactory.createModel(model_type),
            available_models=available_model_types,
            drift_detector=drift_detector_factory(),
        ),
        f"ShadowExpertModel_{model_type}": ShadowExpertModel(
            model_type=model_type,
            drift_detector=drift_detector_factory(),
        ),
        # f"Standard_{model_type}": ModelFactory.createModel(model_type),
    }


def initialize_results(models):
    return {
        name: {
            "accuracy_series": [],
            "drifts": [],
            "period_correct": 0,
            "total_correct": 0,
            "period_total": 0,
            "total_instances": 0,
        }
        for name in models
    }


def learn_and_track_event(model, result, x, y, instance_count, track_events):
    if isinstance(model, (ActiveExpertRepository, ActiveExpertRepositoryMultiModel)):
        active_model_before = model.get_active_expert_label()
        repo_size_before = len(model.repository)
        model.learn_one(x, y)
        active_model_after = model.get_active_expert_label()
        repo_size_after = len(model.repository)

        if track_events == "drifts" and repo_size_after > repo_size_before:
            result["drifts"].append(instance_count)
        elif track_events == "model_changes" and active_model_after != active_model_before:
            result["drifts"].append(instance_count)

    elif isinstance(model, ShadowExpertModel):
        active_model_before = model.get_active_expert_label()
        drifts_before = len(model.drifts_detected)
        model.learn_one(x, y)
        active_model_after = model.get_active_expert_label()
        drifts_after = len(model.drifts_detected)

        if track_events == "drifts" and drifts_after > drifts_before:
            result["drifts"].append(instance_count)
        elif track_events == "model_changes" and active_model_after != active_model_before:
            result["drifts"].append(instance_count)

    else:
        model.learn_one(x, y)


def get_active_model_label(model):
    if isinstance(model, (ActiveExpertRepository, ActiveExpertRepositoryMultiModel, ShadowExpertModel)):
        return model.get_active_expert_label()
    return model.__class__.__name__


def write_chunk_results(csv_path, model_type, models, results, chunk_index, stream_label):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        for model_name, model in models.items():
            result = results[model_name]
            period_acc = (
                result["period_correct"] / result["period_total"] * 100
                if result["period_total"] > 0
                else 0
            )
            cum_acc = (
                result["total_correct"] / result["total_instances"] * 100
                if result["total_instances"] > 0
                else 0
            )

            writer.writerow(
                [
                    model_type,
                    model_name,
                    chunk_index,
                    stream_label,
                    get_active_model_label(model),
                    period_acc,
                    cum_acc,
                ]
            )

            result["period_correct"] = 0
            result["period_total"] = 0


def remove_counter_fields(results):
    for result in results.values():
        result.pop("period_correct", None)
        result.pop("total_correct", None)
        result.pop("period_total", None)
        result.pop("total_instances", None)


def evaluate_model_type(
    model_type,
    available_model_types,
    *,
    seed=42,
    chunk_size=500,
    stream_length=2500,
    concepts=None,
    stream_factory=None,
    drift_detector_factory=default_drift_detector,
    track_events="model_changes",
    output_dir=".",
    csv_filename_template="{model_type}_accuracy_results.csv",
    plot_filename_template="sliding_accuracy_{model_type}.png",
    plot_window=100,
    known_drift=None,
    show_plot_inline=False,
):
    if track_events not in {"drifts", "model_changes"}:
        raise ValueError("track_events must be either 'drifts' or 'model_changes'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / csv_filename_template.format(model_type=model_type)
    plot_path = output_dir / plot_filename_template.format(model_type=model_type)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

    print("\n=========================================")
    print(f"Evaluating Model Type: {model_type}")
    print("=========================================")

    if stream_factory is None:
        if concepts is None:
            concepts = default_concepts(seed=seed)
        raw_gen = recurring_stream_generator(concepts, chunk_size=chunk_size)
    else:
        raw_gen = stream_factory()

    my_stream = islice(raw_gen, stream_length)
    models = build_models(model_type, available_model_types, drift_detector_factory)
    results = initialize_results(models)

    print(f"Starting streaming process for {model_type}...")
    for instance_count, (x, y, stream_label) in enumerate(my_stream, start=1):
        for model_name, model in models.items():
            result = results[model_name]
            y_pred = model.predict_one(x)

            is_correct = y_pred == y
            if y_pred is not None:
                result["accuracy_series"].append(100.0 if is_correct else 0.0)
                if is_correct:
                    result["period_correct"] += 1
                    result["total_correct"] += 1
                result["period_total"] += 1
                result["total_instances"] += 1
            else:
                result["accuracy_series"].append(np.nan)

            learn_and_track_event(model, result, x, y, instance_count, track_events)

        if instance_count % chunk_size == 0:
            chunk_index = instance_count // chunk_size
            write_chunk_results(csv_path, model_type, models, results, chunk_index, stream_label)
            print(f"[{model_type}] Processed {instance_count} instances...")

    print(f"Finished evaluating {model_type}. Plotting results...")

    remove_counter_fields(results)

    event_label = "Model change" if track_events == "model_changes" else "Drift"
    plot_sliding_accuracy(
        results,
        window=plot_window,
        show_inline=show_plot_inline,
        save_path=plot_path,
        event_label=event_label,
        known_drift=known_drift,
    )
    print(f"Plot saved to '{plot_path}'")

    return {
        "models": models,
        "results": results,
        "csv_path": csv_path,
        "plot_path": plot_path,
    }


def evaluate_models(
    model_types,
    *,
    seed=42,
    available_model_types=None,
    **evaluation_kwargs,
):
    set_global_seeds(seed)

    if available_model_types is None:
        available_model_types = model_types

    return {
        model_type: evaluate_model_type(
            model_type,
            available_model_types,
            seed=seed,
            **evaluation_kwargs,
        )
        for model_type in model_types
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AER methods on synthetic recurring streams, real River streams, or known-drift streams."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="synthetic_recurring",
        help="Dataset/stream to evaluate.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=MODEL_TYPES,
        help="Model types to evaluate.",
    )
    parser.add_argument(
        "--stream-length",
        type=int,
        default=None,
        help="Override the default number of stream instances.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override the CSV reporting chunk size.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where CSV files and plots are written.",
    )
    parser.add_argument(
        "--river-data-dir",
        default=DEFAULT_RIVER_DATA_DIR,
        help="Writable cache directory for remote River datasets such as Elec2.",
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=100,
        help="Sliding accuracy plot window size.",
    )
    parser.add_argument(
        "--show-plot-inline",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def run_from_cli():
    args = parse_args()
    config = DATASETS[args.dataset]
    stream_length = args.stream_length or config["stream_length"]
    chunk_size = args.chunk_size or config["chunk_size"]
    output_dir = args.output_dir if args.output_dir is not None else config["output_dir"]

    os.environ.setdefault("RIVER_DATA", str(Path(args.river_data_dir).resolve()))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logging.info("Evaluating %s: %s", args.dataset, config["description"])

    filename_prefix = "" if args.dataset == "synthetic_recurring" else f"{args.dataset}_"

    evaluate_models(
        args.model_types,
        stream_factory=config["stream_factory"],
        chunk_size=chunk_size,
        stream_length=stream_length,
        track_events=config["track_events"],
        output_dir=output_dir,
        csv_filename_template=f"{filename_prefix}{{model_type}}_accuracy_results.csv",
        plot_filename_template=f"{filename_prefix}sliding_accuracy_{{model_type}}.png",
        plot_window=args.plot_window,
        known_drift=config["known_drift"],
        show_plot_inline=args.show_plot_inline,
    )


if __name__ == "__main__":
    run_from_cli()
