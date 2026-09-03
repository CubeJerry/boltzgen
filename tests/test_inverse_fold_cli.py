from pathlib import Path

import pytest

from boltzgen.cli import boltzgen as cli


def test_inverse_fold_parser_exposes_fasta_first_contract() -> None:
    args = cli.build_parser().parse_args(
        [
            "inverse-fold",
            "candidate_001.yaml",
            "candidate_002.yaml",
            "--output",
            "sequences.fasta",
            "--num-sequences",
            "3",
            "--avoid",
            "MC",
            "--devices",
            "2",
            "--num-workers",
            "7",
            "--use-kernels",
            "false",
        ]
    )

    assert args.command == "inverse-fold"
    assert args.design_spec == [Path("candidate_001.yaml"), Path("candidate_002.yaml")]
    assert args.output == Path("sequences.fasta")
    assert args.num_sequences == 3
    assert args.avoid == "MC"
    assert args.devices == 2
    assert args.num_workers == 7
    assert args.use_kernels == "false"


def test_directory_inputs_are_non_recursive_and_sorted(tmp_path) -> None:
    inputs = tmp_path / "prepared"
    inputs.mkdir()
    (inputs / "candidate_002.yaml").touch()
    (inputs / "candidate_001.yaml").touch()
    (inputs / "notes.txt").touch()
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "candidate_003.yaml").touch()

    resolved = cli.resolve_inverse_fold_design_specs([inputs])

    assert [path.name for path in resolved] == [
        "candidate_001.yaml",
        "candidate_002.yaml",
    ]


def test_duplicate_source_stems_are_rejected(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "candidate.yaml").touch()
    (second / "candidate.yaml").touch()

    with pytest.raises(ValueError, match="stems must be unique"):
        cli.resolve_inverse_fold_design_specs(
            [first / "candidate.yaml", second / "candidate.yaml"]
        )


def test_avoid_codes_are_canonical_and_deduplicated() -> None:
    assert cli.parse_inverse_fold_avoid("mCc") == ["MET", "CYS"]
    with pytest.raises(ValueError, match="Unknown canonical amino-acid code"):
        cli.parse_inverse_fold_avoid("X")


def test_accelerator_options_resolve_to_requested_runtime(monkeypatch) -> None:
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: (8, 0))

    assert cli.resolve_inverse_fold_accelerator(None, "auto") == (4, True)
    assert cli.resolve_inverse_fold_accelerator(2, "false") == (2, False)


def test_output_cannot_overwrite_checkpoint(tmp_path, monkeypatch) -> None:
    design_spec = tmp_path / "candidate.yaml"
    design_spec.touch()
    checkpoint = tmp_path / "inverse_fold.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    moldir = tmp_path / "mols"
    moldir.mkdir()
    args = cli.build_parser().parse_args(
        [
            "inverse-fold",
            str(design_spec),
            "--output",
            str(checkpoint),
            "--checkpoint",
            str(checkpoint),
            "--moldir",
            str(moldir),
        ]
    )
    monkeypatch.setattr(
        cli, "resolve_inverse_fold_accelerator", lambda devices, kernels: (1, False)
    )

    with pytest.raises(ValueError, match="cannot overwrite"):
        cli.inverse_fold_command(args)

    assert checkpoint.read_bytes() == b"checkpoint"


def test_inverse_fold_command_instantiates_only_one_predict_task(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "candidate_001.yaml"
    second = tmp_path / "candidate_002.yaml"
    first.touch()
    second.touch()
    checkpoint = tmp_path / "inverse_fold.ckpt"
    checkpoint.touch()
    moldir = tmp_path / "mols"
    moldir.mkdir()
    output = tmp_path / "sequences.fasta"
    args = cli.build_parser().parse_args(
        [
            "inverse-fold",
            str(first),
            str(second),
            "--output",
            str(output),
            "--num-sequences",
            "2",
            "--avoid",
            "MC",
            "--checkpoint",
            str(checkpoint),
            "--moldir",
            str(moldir),
            "--devices",
            "2",
            "--num-workers",
            "7",
            "--use-kernels",
            "false",
        ]
    )
    artifact_calls = []
    captured = {}

    class FakeTask:
        def run(self, config) -> None:
            captured["config"] = config
            captured["runtime_dir"] = Path(config.output)
            output.write_text(">simulated\nAA\n")

    def fake_artifact_path(_args, artifact, repo_type="model", verbose=True):
        artifact_calls.append((artifact, repo_type))
        return Path(artifact)

    monkeypatch.setattr(cli, "Task", FakeTask)
    monkeypatch.setattr(
        cli, "resolve_inverse_fold_accelerator", lambda devices, kernels: (2, False)
    )
    monkeypatch.setattr(cli, "get_artifact_path", fake_artifact_path)
    monkeypatch.setattr(cli.hydra.utils, "instantiate", lambda config: FakeTask())
    monkeypatch.setattr(
        cli,
        "BinderDesignPipeline",
        lambda *args, **kwargs: pytest.fail("full pipeline was instantiated"),
    )
    monkeypatch.setattr(
        cli,
        "configure_command",
        lambda args: pytest.fail("pipeline configuration was invoked"),
    )
    monkeypatch.setattr(
        cli,
        "execute_command",
        lambda args: pytest.fail("pipeline execution was invoked"),
    )

    cli.inverse_fold_command(args)

    config = captured["config"]
    assert config._target_ == "boltzgen.task.predict.predict.Predict"
    assert config.data.cfg.yaml_path == [str(first.resolve()), str(second.resolve())]
    assert config.data.cfg.multiplicity == 2
    assert config.data.cfg.allow_reserved_filenames is True
    assert config.data.num_workers == 7
    assert config.checkpoint == str(checkpoint)
    assert config.data.cfg.moldir == str(moldir)
    assert config.trainer.devices == 2
    assert config.trainer.logger is False
    assert config.trainer.enable_checkpointing is False
    assert config.override.use_kernels is False
    assert config.override.inverse_fold_args.inverse_fold_restriction == [
        "MET",
        "CYS",
    ]
    assert config.writer._target_.endswith("InverseFoldFastaWriter")
    assert config.writer.output_path == str(output.resolve())
    assert config.writer.source_ids == ["candidate_001", "candidate_002"]
    assert artifact_calls == [(str(moldir), "dataset"), (str(checkpoint), "model")]
    assert not captured["runtime_dir"].exists()
    assert output.read_text() == ">simulated\nAA\n"


def test_run_command_keeps_existing_configure_execute_flow(monkeypatch) -> None:
    calls = []
    args = cli.build_parser().parse_args(["run", "design.yaml", "--output", "out"])
    monkeypatch.setattr(
        cli, "configure_command", lambda args: calls.append("configure")
    )
    monkeypatch.setattr(cli, "execute_command", lambda args: calls.append("execute"))

    cli.run_command(args)

    assert calls == ["configure", "execute"]
