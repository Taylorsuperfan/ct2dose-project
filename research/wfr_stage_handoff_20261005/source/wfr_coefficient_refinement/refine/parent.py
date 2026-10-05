"""Read the exact completed parent experiment, not a reconstructed look-alike."""
from pathlib import Path
import torch
import numpy as np
from cwfr import common as io
from cwfr.config import Config
from cwfr.model import ConditionalWFR
from .integrity import Receipt, require


class ParentExperiment:
    def __init__(self, parent_work, data):
        self.work = Path(parent_work).resolve()
        self.run = self.work / "runs.local/pilot_seed17"
        self.eval = self.work / "evaluations.local/pilot_seed17_val600"
        self.train_receipt = Receipt(self.run, "conditional_training_completed")
        self.eval_receipt = Receipt(self.eval, "validation_inference_completed")
        self.contract = self.train_receipt.json("contract.json")
        self.eval_contract = self.eval_receipt.json("contract.json")
        self.pointer = self.train_receipt.json("best.json")
        self.train_receipt.file(self.pointer["path"])
        ident = self.contract["identity"]; ev = self.eval_contract["identity"]
        require(ident["source_code"] == io.source_identity(), "The parent cwfr source differs from the saved run.")
        require(ev["source_code"] == ident["source_code"], "Parent evaluation uses another source revision.")
        require(ident["cache_identity"] == ev["cache_identity"] == data.identity, "Parent/cache identity mismatch.")
        require(ev["train_contract"] == io.digest(self.contract), "Evaluation/parent linkage differs.")
        require(ev["checkpoint"] == self.pointer, "Parent evaluation is not using the selected checkpoint.")
        require(self.train_receipt.complete["contract_sha256"] == io.digest(self.contract), "Parent receipt linkage differs.")
        require(self.eval_receipt.complete["contract_sha256"] == io.digest(self.eval_contract), "Evaluation receipt linkage differs.")
        self.cfg = Config(**ident["config"]).validate()
        require(self.cfg.stage == "pilot" and self.cfg.updates == 384, "Expected the completed 384-update pilot.")
        self.selection = data.selection("pilot", self.cfg.monitor_records_per_case, self.cfg.monitor_seed)
        require(ident["selection"] == self.selection, "The saved data selection differs.")
        require(ev["validation_ids"] == self.selection["validation_ids"], "Validation IDs or order differ.")
        require(len(self.selection["train_ids"]) == 192 and len(self.selection["monitor_ids"]) == 40,
                "Expected train192 and the original monitor40.")
        self.data = data
        self.identity = {
            "training_contract": io.digest(self.contract),
            "evaluation_contract": io.digest(self.eval_contract),
            "evaluation_ledger": self.eval_receipt.complete["files_sha256"],
            "checkpoint": self.pointer,
            "source_code": ident["source_code"],
            "cache_identity": data.identity,
            "selection": self.selection,
        }

    def load_model(self, device):
        ck = io.load_checkpoint(self.run, self.pointer)
        require(ck["contract_sha256"] == io.digest(self.contract), "Checkpoint/contract link differs.")
        model = ConditionalWFR(self.cfg.channels, self.cfg.point_hidden).to(device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def prediction(self, sid):
        row = self.data.by_id[sid]
        require(row["split"] == "validation", "Saved parent evaluation only covers validation.")
        prefix = "records.local/" + sid + "/"
        meta = self.eval_receipt.json(prefix + "record.json")
        path = self.eval_receipt.file(prefix + "arrays.npz")
        require(meta["sample_id"] == sid and meta["case_id"] == row["case_id"], "Parent prediction record differs.")
        require(meta["contract_sha256"] == io.digest(self.eval_contract), "Parent prediction contract differs.")
        require(meta["source_arrays_sha256"] == row["arrays_sha256"] and
                meta["source_record_sha256"] == row["record_meta_sha256"], "Parent prediction/source pairing differs.")
        require(io.sha(path) == meta["arrays_sha256"], "Parent prediction bytes differ.")
        with np.load(path, allow_pickle=False) as values:
            result = {k: values[k].copy() for k in
                      ("magnitude_normalized", "soft_sign", "prediction_stored", "raw_prediction_stored")}
        return result, {"record_sha256": io.sha(self.eval / (prefix + "record.json")),
                        "arrays_sha256": meta["arrays_sha256"]}

    def recheck(self):
        self.train_receipt.recheck(); self.eval_receipt.recheck()
        self.data.check_metadata_unchanged()
