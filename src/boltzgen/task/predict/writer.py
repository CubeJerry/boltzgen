import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import gemmi
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from tqdm import tqdm

from boltzgen.data import const
from boltzgen.data.data import (
    Structure,
    convert_ccd,
)
from boltzgen.data.feature.featurizer import (
    res_all_gly,
    res_from_atom14,
    res_from_atom37,
)
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.model.modules.masker import BoltzMasker


class FoldingWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(self, design_dir: str, designfolding: bool = False) -> None:
        super().__init__(write_interval="batch")
        self.designfolding = designfolding
        if design_dir is not None:
            self.init_outdir(design_dir)

    def init_outdir(self, design_dir):
        self.outdir = Path(design_dir) / (
            const.folding_design_dirname
            if self.designfolding
            else const.folding_dirname
        )
        self.refold_cif_dir = Path(design_dir) / (
            const.refold_design_cif_dirname
            if self.designfolding
            else const.refold_cif_dirname
        )
        self.refold_cif_dir.mkdir(parents=True, exist_ok=True)
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.failed = 0

    def write_on_batch_end(  # noqa: PLR0915
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
        sample_id: str = None,
    ) -> None:
        """Write the predictions to disk."""
        pred_dict = {}
        for key, value in prediction.items():
            # check object is tensor
            if key in const.eval_keys:
                pred_dict[key] = value.cpu().numpy()
        np.savez_compressed(self.outdir / f"{batch['id'][0]}.npz", **pred_dict)

        # Get best sample
        confidence = 0.8 * pred_dict["iptm"] + 0.2 * pred_dict["ptm"]
        best_idx = np.argmax(confidence)
        best_sample_coords = pred_dict["coords"][best_idx]

        prediction_out = {}
        for k in prediction:
            if k == "coords":
                prediction_out[k] = torch.from_numpy(best_sample_coords)
            else:
                prediction_out[k] = prediction[k][0]

        # Write structure
        structure, _, _ = Structure.from_feat(prediction_out)
        plddt_atom = (
            prediction_out["atom_to_token"].float() @ prediction_out["plddt"].float()
        )
        structure.atoms["bfactor"] = (
            plddt_atom[prediction_out["atom_pad_mask"].bool()].float().cpu().numpy()
        )
        cif_text = to_mmcif(structure)
        open(self.refold_cif_dir / f"{batch['id'][0]}.cif", "w").write(cif_text)

        # Failed prediction handling
        if isinstance(prediction["exception"], bool):
            if prediction["exception"]:
                self.failed += 1
        elif isinstance(prediction["exception"], list):
            if prediction["exception"][0]:
                self.failed += 1

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        print(f"Number of failed structure predictions: {self.failed}")  # noqa: T201


class AffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        design_dir: str,
    ) -> None:
        super().__init__(write_interval="batch")
        if design_dir is not None:
            self.init_outdir(design_dir)

    def init_outdir(self, design_dir):
        self.outdir = Path(design_dir) / const.affinity_dirname
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.failed = 0

    def write_on_batch_end(  # noqa: PLR0915
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
        sample_id: str = None,
    ) -> None:
        """Write the predictions to disk."""
        pred_dict = {}
        for key, value in prediction.items():
            # check object is tensor
            if key in const.eval_keys:
                pred_dict[key] = value.cpu().numpy()
        np.savez_compressed(self.outdir / f"{batch['id'][0]}.npz", **pred_dict)

        if isinstance(prediction["exception"], bool):
            if prediction["exception"]:
                self.failed += 1
        elif isinstance(prediction["exception"], list):
            if prediction["exception"][0]:
                self.failed += 1

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        print(f"Number of failed affinity predictions: {self.failed}")  # noqa: T201


def reconstruct_inverse_fold_sequence(
    original_token_ids: Sequence[int],
    predicted_token_ids: Sequence[int],
    design_mask: Sequence[bool],
    token_pad_mask: Sequence[bool],
    asym_ids: Sequence[int],
    mol_types: Sequence[int],
    token_to_res: Sequence[int],
    token_resolved_mask: Sequence[bool] | None = None,
) -> str:
    """Reconstruct the complete designed protein chain as a sequence."""
    fields = {
        "original_token_ids": original_token_ids,
        "predicted_token_ids": predicted_token_ids,
        "design_mask": design_mask,
        "token_pad_mask": token_pad_mask,
        "asym_ids": asym_ids,
        "mol_types": mol_types,
        "token_to_res": token_to_res,
    }
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inverse-fold token fields have different lengths: {lengths}")
    if token_resolved_mask is not None and len(token_resolved_mask) != len(design_mask):
        raise ValueError("token_resolved_mask has a different length from design_mask")

    protein_id = const.chain_type_ids["PROTEIN"]
    designed_indices = [
        index
        for index, is_designed in enumerate(design_mask)
        if is_designed and token_pad_mask[index] and mol_types[index] == protein_id
    ]
    if not designed_indices:
        raise ValueError("No designed protein residues were found")

    if token_resolved_mask is not None:
        unresolved = [
            index for index in designed_indices if not token_resolved_mask[index]
        ]
        if unresolved:
            raise ValueError(
                "Designed protein residues are unresolved at token indices "
                f"{unresolved}"
            )

    designed_chain_ids = {asym_ids[index] for index in designed_indices}
    if len(designed_chain_ids) != 1:
        raise ValueError(
            "FASTA inverse folding requires exactly one designed protein chain; "
            f"found {len(designed_chain_ids)}"
        )
    designed_chain_id = next(iter(designed_chain_ids))

    merged_token_ids = list(original_token_ids)
    for index in designed_indices:
        predicted_token = const.tokens[predicted_token_ids[index]]
        if predicted_token not in const.canonical_tokens:
            raise ValueError(
                f"Inverse fold predicted non-canonical token {predicted_token!r} "
                f"at token index {index}"
            )
        merged_token_ids[index] = predicted_token_ids[index]

    sequence = []
    seen_residues = set()
    for index, token_id in enumerate(merged_token_ids):
        if (
            not token_pad_mask[index]
            or mol_types[index] != protein_id
            or asym_ids[index] != designed_chain_id
        ):
            continue
        residue_key = (asym_ids[index], token_to_res[index])
        if residue_key in seen_residues:
            continue
        seen_residues.add(residue_key)
        token = const.tokens[token_id]
        try:
            sequence.append(const.prot_token_to_letter[token])
        except KeyError as exc:
            raise ValueError(
                f"Cannot represent protein token {token!r} in FASTA"
            ) from exc

    if not sequence:
        raise ValueError("The designed protein chain is empty")
    return "".join(sequence)


class InverseFoldFastaWriter(BasePredictionWriter):
    """Write inverse-folded binder sequences directly from model predictions."""

    def __init__(
        self,
        output_path: str,
        source_ids: Sequence[str],
        num_sequences: int = 1,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_path = Path(output_path)
        self.source_ids = list(source_ids)
        if num_sequences < 1:
            raise ValueError("num_sequences must be at least 1")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        self.source_order = {
            source_id: index for index, source_id in enumerate(self.source_ids)
        }
        self.num_sequences = num_sequences
        self.records: list[tuple[str, int, str]] = []
        self.failures: list[str] = []

    def write_on_batch_end(
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Collect one sequence record from an inverse-fold prediction."""
        source_id = str(batch["id"][0])
        if prediction.get("exception", False) or prediction.get("skip", False):
            self.failures.append(source_id)
            return

        mask_key = (
            "inverse_fold_design_mask"
            if "inverse_fold_design_mask" in batch
            else "design_mask"
        )
        resolved_mask = (
            batch["token_resolved_mask"][0].detach().cpu().bool().tolist()
            if "token_resolved_mask" in batch
            else None
        )
        try:
            sequence = reconstruct_inverse_fold_sequence(
                original_token_ids=torch.argmax(batch["res_type"][0], dim=-1)
                .detach()
                .cpu()
                .tolist(),
                predicted_token_ids=torch.argmax(
                    prediction["res_type"][0], dim=-1
                )
                .detach()
                .cpu()
                .tolist(),
                design_mask=batch[mask_key][0].detach().cpu().bool().tolist(),
                token_pad_mask=batch["token_pad_mask"][0]
                .detach()
                .cpu()
                .bool()
                .tolist(),
                asym_ids=batch["asym_id"][0].detach().cpu().tolist(),
                mol_types=batch["mol_type"][0].detach().cpu().tolist(),
                token_to_res=batch["token_to_res"][0].detach().cpu().tolist(),
                token_resolved_mask=resolved_mask,
            )
        except (KeyError, ValueError) as exc:
            self.failures.append(f"{source_id}: {exc}")
            return
        sample_idx = (
            int(batch["data_sample_idx"][0])
            if "data_sample_idx" in batch
            else 0
        )
        self.records.append((source_id, sample_idx, sequence))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Gather records across ranks and atomically write the final FASTA."""
        payloads = [{"records": self.records, "failures": self.failures}]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payloads = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                payloads,
                {"records": self.records, "failures": self.failures},
            )

        records = {}
        failures = []
        for payload in payloads:
            failures.extend(payload["failures"])
            for source_id, sample_idx, sequence in payload["records"]:
                records.setdefault((source_id, sample_idx), sequence)

        expected_keys = {
            (source_id, sample_idx)
            for source_id in self.source_ids
            for sample_idx in range(self.num_sequences)
        }
        if set(records) != expected_keys:
            raise RuntimeError(
                f"Inverse folding produced {len(records)} of {len(expected_keys)} "
                "expected sequences; "
                f"missing: {sorted(expected_keys - set(records))}; "
                f"unexpected: {sorted(set(records) - expected_keys)}; "
                f"failures: {sorted(set(failures))}"
            )
        if not trainer.is_global_zero:
            return

        width = max(3, len(str(self.num_sequences - 1)))
        output = []
        for (source_id, sample_idx), sequence in sorted(
            records.items(),
            key=lambda item: (
                self.source_order[item[0][0]],
                item[0][1],
            ),
        ):
            record_id = (
                source_id
                if self.num_sequences == 1
                else f"{source_id}__if{sample_idx:0{width}d}"
            )
            output.append(f">{record_id}\n{sequence}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_name(f".{self.output_path.name}.tmp")
        temporary_path.write_text("\n".join(output) + "\n")
        temporary_path.replace(self.output_path)
        print(f"Wrote {len(records)} inverse-folded sequences to {self.output_path}")


class DesignWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        output_dir: str,
        res_atoms_only: bool,
        save_traj: bool = False,
        save_x0_traj: bool = False,
        atom14: bool = True,
        atom37: bool = False,
        backbone_only: bool = False,
        inverse_fold: bool = False,
        file_suffix: str = "",
        write_native: bool = True,
        design: bool = True,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.mol_dir = Path(output_dir) / const.molecules_dirname
        self.mol_dir.mkdir(parents=True, exist_ok=True)
        self.save_traj = save_traj
        self.save_x0_traj = save_x0_traj
        self.res_atoms_only = res_atoms_only
        self.file_suffix = file_suffix
        self.failed = 0
        self.write_native = write_native
        self.design = design

        # Create the output directories
        self.atom14 = atom14
        self.atom37 = atom37
        self.inverse_fold = inverse_fold
        self.backbone_only = backbone_only
        self.used_stems = set()
        self.init_outdir(output_dir)

    def init_outdir(self, outdir):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(  # noqa: PLR0915
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
        sample_id: str = None,
    ) -> None:
        if prediction["exception"]:
            self.failed += 1
            return
        n_samples, _, _ = prediction["coords"].shape

        # TODO: remove this which is only here for temporary backward compatibility
        masker = BoltzMasker(mask=True, mask_backbone=False)
        feat_masked = masker(batch)
        prediction["ref_element"] = feat_masked["ref_element"]
        prediction["ref_atom_name_chars"] = feat_masked["ref_atom_name_chars"]
        """Write the predictions to disk."""
        # Check for extra molecules
        if batch["extra_mols"] is not None:
            extra_mols = batch["extra_mols"][0]
            for k, v in extra_mols.items():
                with open(self.mol_dir / f"{k}.pkl", "wb") as f:
                    pickle.dump(v, f)

        # write samples to disk
        for n in range(n_samples):
            # get structure for all generated coords
            sample, native = {}, {}

            for k in set(prediction.keys()) & set(batch.keys()):
                if k == "coords":
                    native[k] = batch[k][0][0].unsqueeze(0)
                    sample[k] = prediction[k][n]
                    
                if k in const.token_features:
                    sample[k] = prediction[k][0]
                    native[k] = batch[k][0]
                elif k in const.atom_features:
                    if k == "coords":
                        native[k] = batch[k][0][0].unsqueeze(0)
                        sample[k] = prediction[k][n]
                    else:
                        native[k] = batch[k][0]
                        sample[k] = prediction[k][0]
                elif k == "exception":
                    sample[k] = prediction[k]
                    native[k] = batch[k]
                else:
                    native[k] = batch[k][0]
                    sample[k] = prediction[k][0]
                    native[k] = batch[k][0]

            if self.atom14:
                sample = res_from_atom14(sample)
            elif self.atom37:
                sample = res_from_atom37(sample)
            elif self.backbone_only:
                sample = res_all_gly(sample)


            design_mask = batch["design_mask"][0].bool()
            assert design_mask.sum() == sample["design_mask"].sum()

            if self.inverse_fold:
                token_ids = torch.argmax(sample["res_type"], dim=-1)
                tokens = [const.tokens[i] for i in token_ids]
                ccds = [convert_ccd(token) for token in tokens]

                ccds = torch.tensor(ccds).to(sample["res_type"])
                sample["ccd"][design_mask] = ccds[design_mask]

            try:
                structure, _, _ = Structure.from_feat(sample)
                str_native, _, _ = Structure.from_feat(native)

                # write structure to cif
                if sample_id is not None:
                    file_name = f"{sample_id}_{n}{self.file_suffix}"
                else:
                    stem = str(batch["id"][0])
                    multiplicity = getattr(trainer.datamodule.cfg, "multiplicity", 1)
                    total_files = multiplicity * n_samples
                    sample_idx = (
                        int(batch["data_sample_idx"][0])
                        if "data_sample_idx" in batch
                        else 0
                    )
                    global_idx = sample_idx * n_samples + n

                    if total_files > 1:
                        num_digits = len(str(total_files - 1))
                        file_name = (
                            f"{stem}_{global_idx:0{num_digits}d}{self.file_suffix}"
                        )
                    else:
                        file_name = f"{stem}{self.file_suffix}"

                native_path = f"{self.outdir}/{file_name}_native.cif"
                gen_path = f"{self.outdir}/{file_name}.cif"

                # design mask bfactor
                design_mask = batch["design_mask"][0].float()
                atom_design_mask = (
                    sample["atom_to_token"].float() @ design_mask.unsqueeze(-1).float()
                )
                design_mask = native["design_mask"].float()

                atom_design_mask = atom_design_mask.squeeze().bool()
                bfactor = atom_design_mask * 100

                # binding type bfactor
                binding_type = batch["binding_type"][0].float()
                atom_binding_type = (
                    sample["atom_to_token"].float() @ binding_type.unsqueeze(-1).float()
                )

                atom_binding_type = atom_binding_type.squeeze().int()
                bfactor[atom_binding_type == const.binding_type_ids["BINDING"]] = 60

                bfactor = atom_design_mask[sample["atom_pad_mask"].bool()].float()
                str_native.atoms["bfactor"] = bfactor.cpu().numpy()
                structure.atoms["bfactor"] = bfactor.cpu().numpy()

                # Add dummy (0-coord) design side chains if inverse fold
                if self.inverse_fold:
                    atom_design_mask_no_pad = atom_design_mask[
                        native["atom_pad_mask"].bool()
                    ]
                    res_design_mask = np.array(
                        [
                            all(
                                atom_design_mask_no_pad[
                                    res["atom_idx"] : res["atom_idx"] + res["atom_num"]
                                ]
                            )
                            for res in structure.residues
                        ]
                    )
                    structure = Structure.add_side_chains(
                        structure, residue_mask=res_design_mask
                    )

                if self.write_native:
                    open(native_path, "w").write(to_mmcif(str_native))

                pred_binding_mask = prediction["binding_type"][0].cpu().bool().numpy()
                if self.design:
                    chain_design_mask = (
                        prediction["chain_design_mask"][0].cpu().bool().numpy()
                    )
                pred_design_mask = prediction["design_mask"][0].cpu().bool().numpy()
                design_color_features = np.ones_like(pred_binding_mask) * 0.8
                design_color_features[pred_binding_mask] = 1.0
                if self.design:
                    design_color_features[chain_design_mask] = 0.0
                design_color_features[pred_design_mask] = 0.6

                # Create a mask to identify unique token-to-res mappings.
                # This is for small molecules where multiple tokens can be mapped to the same residue.
                token_to_res = prediction["token_to_res"][0].cpu().numpy()
                unique_mask = np.ones_like(token_to_res, dtype=bool)
                unique_mask[1:] = token_to_res[1:] != token_to_res[:-1]
                design_color_features = design_color_features[unique_mask]
                open(gen_path, "w").write(
                    to_mmcif(
                        structure,
                        design_coloring=True,
                        color_features=design_color_features,
                    )
                )

                # Write metadata
                metadata_path = f"{self.outdir}/{file_name}.npz"
                token_mask = sample["token_pad_mask"].bool()

                # Build metadata dict with required fields
                metadata_dict = {
                    "design_mask": design_mask[token_mask].cpu().numpy(),
                    "mol_type": sample["mol_type"][token_mask].cpu().numpy(),
                    "ss_type": sample["ss_type"][token_mask].cpu().numpy(),
                    "token_resolved_mask": sample["token_resolved_mask"][token_mask].cpu().numpy(),
                    "binding_type": binding_type[token_mask].cpu().numpy(),
                }

                # Add optional fields only if they have valid values (avoid None -> object array)
                if "inverse_fold_design_mask" in sample:
                    metadata_dict["inverse_fold_design_mask"] = (
                        sample["inverse_fold_design_mask"][token_mask].cpu().numpy()
                    )

                # Per-residue amino acid constraints (for inverse folding step)
                # Only save if constraints exist AND have non-zero values
                if "aa_constraint_mask" in batch:
                    aa_mask = batch["aa_constraint_mask"][0]
                    if aa_mask.any():  # Only save if there are actual constraints
                        metadata_dict["aa_constraint_mask"] = aa_mask[token_mask].cpu().numpy()

                np.savez_compressed(metadata_path, **metadata_dict)

                # Write trajectories
                if self.save_traj:
                    trajs = torch.stack(prediction["coords_traj"], dim=1)
                    traj = trajs[n]
                    aligned = [traj[0]]
                    for frame in traj[1:]:
                        with torch.autocast("cuda", enabled=False):
                            aligned.append(
                                weighted_rigid_align(
                                    frame.float().unsqueeze(0),
                                    aligned[-1].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                )
                                .to(frame)
                                .squeeze()
                            )

                    mmcifs = []
                    for _idx, frame in tqdm(
                        enumerate(aligned), desc="Writing traj.", total=len(aligned)
                    ):
                        sample["coords"] = frame
                        if self.atom14:
                            sample = res_from_atom14(sample)
                        elif self.atom37:
                            sample = res_from_atom37(sample)
                        else:
                            raise ValueError("Either atom14 or atom37 must be true")

                        str_frame, _, _ = Structure.from_feat(sample)
                        mmcifs.append(to_mmcif(str_frame))

                    open(self.outdir / f"{file_name}_traj.cif", "w").write(
                        self.combine_mmcif_models(mmcifs)
                    )

                # Write x0 trajectories
                if self.save_x0_traj:
                    trajs = torch.stack(prediction["x0_coords_traj"], dim=1)
                    traj = trajs[n]
                    aligned = [traj[0]]
                    for frame in traj[1:]:
                        with torch.autocast("cuda", enabled=False):
                            aligned.append(
                                weighted_rigid_align(
                                    frame.float().unsqueeze(0),
                                    aligned[-1].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                )
                                .to(frame)
                                .squeeze()
                            )

                    mmcifs = []
                    for _idx, frame in tqdm(
                        enumerate(aligned), desc="Writing x0 traj.", total=len(aligned)
                    ):
                        sample["coords"] = frame
                        if self.atom14:
                            sample = res_from_atom14(sample)
                        elif self.atom37:
                            sample = res_from_atom37(sample)
                        else:
                            raise ValueError("Either atom14 or atom37 must be true")

                        str_frame, _, _ = Structure.from_feat(sample)
                        mmcifs.append(to_mmcif(str_frame))

                    open(self.outdir / f"{file_name}_x0_traj.cif", "w").write(
                        self.combine_mmcif_models(mmcifs)
                    )

            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()  # noqa: T201
                msg = f"predict/writer.py: Validation structure writing failed on {batch['id'][0]} with error {e}. Skipping."
                print(msg)

    def combine_mmcif_models(self, mmcif_strings):
        gemmi_structure = None
        for model_number, mmcif_string in enumerate(mmcif_strings, start=1):
            block = gemmi.cif.read_string(mmcif_string).sole_block()
            frame_structure = gemmi.make_structure_from_block(block)
            gemmi_model = frame_structure[0]
            try:
                gemmi_model.num = model_number
            except AttributeError:
                gemmi_model.name = str(model_number)

            if gemmi_structure is None:
                gemmi_structure = frame_structure
                gemmi_structure.name = "trajectory"
            else:
                gemmi_structure.add_model(gemmi_model)

        if gemmi_structure is None:
            raise ValueError("At least one mmCIF model is required")

        return gemmi_structure.make_mmcif_document().as_string()

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
