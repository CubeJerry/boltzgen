from types import SimpleNamespace

import pytest

from boltzgen.data import const
from boltzgen.task.predict import writer as writer_module
from boltzgen.task.predict.data_from_yaml import validate_yaml_filenames
from boltzgen.task.predict.writer import (
    InverseFoldFastaWriter,
    reconstruct_inverse_fold_sequence,
)


def _token_ids(sequence: str) -> list[int]:
    return [const.token_ids[const.prot_letter_to_token[letter]] for letter in sequence]


class FakeTensor:
    """Minimal tensor surface needed by the prediction writer tests."""

    def __init__(self, value: object) -> None:
        self.value = value

    def __getitem__(self, index: int) -> "FakeTensor":
        return FakeTensor(self.value[index])

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def bool(self) -> "FakeTensor":
        return self

    def tolist(self) -> object:
        return self.value


def test_reconstruction_preserves_fixed_residues_and_excludes_target() -> None:
    target = _token_ids("DDD")
    binder = _token_ids("GCS")
    original = target + binder
    predicted = _token_ids("AAA") + _token_ids("VMR")

    sequence = reconstruct_inverse_fold_sequence(
        original_token_ids=original,
        predicted_token_ids=predicted,
        design_mask=[False, False, False, False, True, False],
        token_pad_mask=[True] * 6,
        asym_ids=[1, 1, 1, 7, 7, 7],
        mol_types=[const.chain_type_ids["PROTEIN"]] * 6,
        token_to_res=list(range(6)),
        token_resolved_mask=[True] * 6,
    )

    assert sequence == "GMS"


def test_reconstruction_rejects_multiple_designed_protein_chains() -> None:
    with pytest.raises(ValueError, match="exactly one designed protein chain"):
        reconstruct_inverse_fold_sequence(
            original_token_ids=_token_ids("GC"),
            predicted_token_ids=_token_ids("AR"),
            design_mask=[True, True],
            token_pad_mask=[True, True],
            asym_ids=[4, 9],
            mol_types=[const.chain_type_ids["PROTEIN"]] * 2,
            token_to_res=[0, 1],
        )


def test_reconstruction_rejects_unresolved_designed_residue() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        reconstruct_inverse_fold_sequence(
            original_token_ids=_token_ids("GC"),
            predicted_token_ids=_token_ids("AR"),
            design_mask=[False, True],
            token_pad_mask=[True, True],
            asym_ids=[7, 7],
            mol_types=[const.chain_type_ids["PROTEIN"]] * 2,
            token_to_res=[0, 1],
            token_resolved_mask=[True, False],
        )


def test_writer_uses_inverse_fold_mask_and_writes_complete_chain(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "sequences.fasta"
    writer = InverseFoldFastaWriter(
        output_path=str(output), source_ids=["candidate_001"]
    )
    protein = const.chain_type_ids["PROTEIN"]
    original = _token_ids("DDDGCS")
    predicted = _token_ids("AAAVMR")
    batch = {
        "id": ["candidate_001"],
        "res_type": FakeTensor([original]),
        "design_mask": FakeTensor([[False, False, False, True, True, True]]),
        "inverse_fold_design_mask": FakeTensor(
            [[False, False, False, False, True, False]]
        ),
        "token_pad_mask": FakeTensor([[True] * 6]),
        "token_resolved_mask": FakeTensor([[True] * 6]),
        "asym_id": FakeTensor([[1, 1, 1, 7, 7, 7]]),
        "mol_type": FakeTensor([[protein] * 6]),
        "token_to_res": FakeTensor([list(range(6))]),
    }
    prediction = {"exception": False, "res_type": FakeTensor([predicted])}

    monkeypatch.setattr(writer_module.torch, "argmax", lambda value, dim: value)
    monkeypatch.setattr(writer_module.torch.distributed, "is_available", lambda: False)

    writer.write_on_batch_end(prediction=prediction, batch=batch)
    writer.on_predict_epoch_end(SimpleNamespace(is_global_zero=True), None)

    assert output.read_text() == ">candidate_001\nGMS\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sequences.fasta"]


def test_writer_orders_sources_and_numbers_multiple_sequences(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "sequences.fasta"
    writer = InverseFoldFastaWriter(
        output_path=str(output),
        source_ids=["candidate_001", "candidate_002"],
        num_sequences=2,
    )
    writer.records = [
        ("candidate_002", 1, "VV"),
        ("candidate_001", 1, "RR"),
        ("candidate_002", 0, "GG"),
        ("candidate_001", 0, "AA"),
    ]
    monkeypatch.setattr(writer_module.torch.distributed, "is_available", lambda: False)

    writer.on_predict_epoch_end(SimpleNamespace(is_global_zero=True), None)

    assert output.read_text() == (
        ">candidate_001__if000\nAA\n"
        ">candidate_001__if001\nRR\n"
        ">candidate_002__if000\nGG\n"
        ">candidate_002__if001\nVV\n"
    )


def test_writer_maps_one_record_per_source_without_suffix(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "sequences.fasta"
    writer = InverseFoldFastaWriter(
        output_path=str(output),
        source_ids=["candidate_001", "candidate_002"],
    )
    writer.records = [
        ("candidate_002", 0, "VV"),
        ("candidate_001", 0, "AA"),
    ]
    monkeypatch.setattr(writer_module.torch.distributed, "is_available", lambda: False)

    writer.on_predict_epoch_end(SimpleNamespace(is_global_zero=True), None)

    assert output.read_text() == (">candidate_001\nAA\n>candidate_002\nVV\n")


def test_writer_gathers_rank_records_and_deduplicates_padding(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "sequences.fasta"
    writer = InverseFoldFastaWriter(
        output_path=str(output),
        source_ids=["candidate_001", "candidate_002"],
    )
    writer.records = [("candidate_001", 0, "AA")]
    distributed = writer_module.torch.distributed
    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)

    def fake_all_gather(payloads, local_payload) -> None:
        payloads[:] = [
            local_payload,
            {
                "records": [
                    ("candidate_001", 0, "AA"),
                    ("candidate_002", 0, "VV"),
                ],
                "failures": [],
            },
        ]

    monkeypatch.setattr(distributed, "all_gather_object", fake_all_gather)

    writer.on_predict_epoch_end(SimpleNamespace(is_global_zero=True), None)

    assert output.read_text() == (">candidate_001\nAA\n>candidate_002\nVV\n")


def test_reserved_yaml_names_are_scoped_to_fasta_mode(tmp_path) -> None:
    candidate = tmp_path / "candidate_001.yaml"
    candidate.touch()

    with pytest.raises(ValueError, match="reserved for internal file indexing"):
        validate_yaml_filenames([str(candidate)])

    validate_yaml_filenames([str(candidate)], allow_reserved_filenames=True)
