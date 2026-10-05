"""CPU unit tests use manufactured arrays, never user medical data."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from cwfr import common as io
from cwfr.model import ConditionalWFR
from refine.config import RefinementConfig, GUARDS
from refine.integrity import save_npz, save_weights, Receipt, state_digest
from refine.model import CoefficientHead, predict_new_record
from refine.losses import components, objective
from refine.selection import assess, improves_primary
from refine.train import make_schedule, summarize_rows, factor_metrics


class CoreTests(unittest.TestCase):
    def setUp(self):
        io.seed_all(3)
        self.cfg = RefinementConfig()

    def batch(self, shape=(1, 4, 4, 4)):
        u = torch.linspace(-1, 1, np.prod(shape)).reshape(shape)
        a = torch.ones_like(u) * 1.5
        b = torch.ones_like(u) * 2
        return {"magnitude": a, "residual": u, "base": b, "target": b + .1*u,
                "condition": torch.zeros(shape[0], 2, *shape[1:]), "scale": .1}

    def test_config_validation(self):
        self.cfg.validate()
        for field, value in [("updates", 0), ("learning_rate", float("nan")),
                             ("guard_relative_tolerance", -1), ("profile_weight", 0),
                             ("profile_relative_share", 2)]:
            with self.assertRaises(ValueError): replace(self.cfg, **{field: value}).validate()

    def test_no_oracle_training_setting(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, uncertainty_oracle_used_for_training=True).validate()

    def test_initial_head_matches_parent(self):
        parent = ConditionalWFR(2, 4)
        with torch.no_grad(): parent.sign_out.bias.fill_(.4)
        head = CoefficientHead.from_parent(parent)
        x = torch.randn(1, 2, 4, 4, 4)
        torch.testing.assert_close(head(x), parent.sign_logits(x)[:, 0], rtol=0, atol=0)

    def test_head_is_a_deep_copy(self):
        parent = ConditionalWFR(2, 4); head = CoefficientHead.from_parent(parent)
        before = state_digest(parent.state_dict())
        with torch.no_grad(): head.out.bias.add_(1)
        self.assertEqual(state_digest(parent.state_dict()), before)

    def test_transport_remains_unchanged_when_training_head(self):
        parent = ConditionalWFR(2, 4); before = state_digest(parent.state_dict())
        head = CoefficientHead.from_parent(parent); opt = torch.optim.SGD(head.parameters(), .1)
        opt.zero_grad(); head(torch.randn(1, 2, 4, 4, 4)).square().mean().backward(); opt.step()
        self.assertEqual(before, state_digest(parent.state_dict()))

    def test_bounded_output(self):
        head = CoefficientHead(2)
        out = head.coefficient(torch.randn(1, 2, 4, 4, 4))
        self.assertTrue(bool((out.abs() <= 1).all()))

    def test_target_free_interface(self):
        names = inspect.signature(predict_new_record).parameters
        for name in names:
            self.assertNotIn("target", name)
            self.assertNotIn("truth", name)

    def test_real_parent_and_head_prediction(self):
        parent = ConditionalWFR(2, 4); head = CoefficientHead.from_parent(parent)
        x = torch.zeros(1, 2, 4, 4, 4); base = np.ones((4, 4, 4))
        out = predict_new_record(parent, head, x, base, .1, 1000., steps=1, query_chunk=64)
        np.testing.assert_allclose(out["prediction_stored"], .001)
        self.assertFalse(out["target_used_at_inference"])

    def test_objective_arms_differ_only_profile(self):
        batch = self.batch(); logits = torch.zeros_like(batch["residual"], requires_grad=True)
        normalizers = {k: 1. for k in ("sign_bce", "reconstruction_mse", "profile_relative", "profile_absolute_mse")}
        a, terms_a = objective(logits, batch, normalizers, self.cfg, "composition")
        b, terms_b = objective(logits, batch, normalizers, self.cfg, "profile")
        self.assertAlmostEqual(float(b-a), self.cfg.profile_weight * terms_b["profile_scaled"], places=6)
        self.assertEqual(terms_a["sign_scaled"], terms_b["sign_scaled"])

    def test_objective_has_coefficient_gradient_only(self):
        b = self.batch(); logits = torch.zeros_like(b["residual"], requires_grad=True)
        norm = {k: 1. for k in ("sign_bce", "reconstruction_mse", "profile_relative", "profile_absolute_mse")}
        loss, _ = objective(logits, b, norm, self.cfg, "profile"); loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertIsNone(b["magnitude"].grad)

    def test_reject_trainable_magnitude(self):
        b = self.batch(); b["magnitude"].requires_grad_(True)
        with self.assertRaises(ValueError):
            components(torch.zeros_like(b["residual"]), b["magnitude"], b["residual"], b["base"], b["target"], .1, self.cfg)

    def test_zero_target_losses_are_finite(self):
        u = torch.zeros(1, 4, 4, 4)
        p = components(u, torch.ones_like(u), u, u, u, .1, self.cfg)
        for value in p.values(): self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual(float(p["sign_batch"]), 0.)

    def test_profile_uses_array_x(self):
        shape = (1, 3, 4, 5); target = torch.ones(shape)
        target[0, 1, 2, 3] = 10
        u = torch.zeros(shape); a = torch.ones(shape); logits = torch.zeros(shape)
        logits[0, 1, 2, :] = 1
        p = components(logits, a, u, target, target, 1., self.cfg)
        self.assertAlmostEqual(float(p["profile_absolute_mse"]), float(torch.tanh(torch.tensor(.5)).square()), places=6)

    def test_profile_relative_matches_manual(self):
        b = self.batch(); logits = torch.zeros_like(b["residual"])
        p = components(logits, b["magnitude"], b["residual"], b["base"], b["target"], b["scale"], self.cfg)
        index = np.unravel_index(int(b["target"][0].argmax()), (4, 4, 4))
        gt = b["target"][0, index[0], index[1]]
        e = .1*b["residual"][0, index[0], index[1]]
        self.assertAlmostEqual(float(p["profile_relative"]), float((e.abs()/(gt.abs()+1e-8)).mean()), places=7)

    def test_sign_weighting_preserves_batch_normalization(self):
        b = self.batch((2, 4, 4, 4)); b["residual"][1] *= 4
        logits = torch.ones_like(b["residual"])
        p = components(logits, b["magnitude"], b["residual"], b["base"], b["target"], .1, self.cfg)
        self.assertAlmostEqual(float(p["sign_batch"]), float(p["sign_numerator"].sum()/p["sign_denominator"].sum()), places=6)

    def test_no_input_mutation(self):
        b = self.batch(); before = {k: v.clone() for k, v in b.items() if torch.is_tensor(v)}
        norm = {k: 1. for k in ("sign_bce", "reconstruction_mse", "profile_relative", "profile_absolute_mse")}
        objective(torch.zeros_like(b["residual"]), b, norm, self.cfg, "profile")
        for k, v in before.items(): torch.testing.assert_close(b[k], v, rtol=0, atol=0)

    def test_schedule_same_across_arms_and_resume(self):
        a = make_schedule(["a", "b", "c"], self.cfg)
        b = make_schedule(["a", "b", "c"], self.cfg)
        self.assertEqual(a, b); self.assertEqual(a[128:], b[128:])

    def test_schedule_seed_matters(self):
        self.assertNotEqual(make_schedule(["a", "b", "c"], self.cfg),
                            make_schedule(["a", "b", "c"], replace(self.cfg, seed=30)))

    def refs(self):
        row = {k: 1. for k in GUARDS}; row["x_mean_pct"] = 10.
        return row, {"a": dict(row), "b": dict(row)}

    def test_selection_rejects_global_only_improvement(self):
        ref, cases = self.refs(); cur = dict(ref, rmse_stored_units=.8, x_mean_pct=11)
        self.assertFalse(assess(cur, cases, ref, cases, self.cfg)["eligible"])

    def test_selection_enforces_each_case(self):
        ref, cases = self.refs(); new = copy.deepcopy(cases)
        new["a"]["y_mean_pct"] = 1.1
        self.assertFalse(assess(ref, new, ref, cases, self.cfg)["eligible"])

    def test_selection_allows_declared_margin_only(self):
        ref, cases = self.refs(); cur = dict(ref, rmse_stored_units=1.005, x_mean_pct=9.)
        self.assertTrue(assess(cur, cases, ref, cases, self.cfg)["eligible"])
        cur["rmse_stored_units"] = 1.02
        self.assertFalse(assess(cur, cases, ref, cases, self.cfg)["eligible"])

    def test_selection_rejects_missing_metric(self):
        ref, cases = self.refs(); cur = dict(ref, y_mean_pct=None)
        self.assertFalse(assess(cur, cases, ref, cases, self.cfg)["eligible"])

    def test_selection_keeps_parent_on_tie(self):
        ref, _ = self.refs(); self.assertFalse(improves_primary(ref, ref, self.cfg))
        self.assertTrue(improves_primary(dict(ref, x_mean_pct=9), ref, self.cfg))

    def test_selection_mismatched_cases(self):
        ref, cases = self.refs()
        self.assertFalse(assess(ref, {"a": ref}, ref, cases, self.cfg)["eligible"])

    def test_equal_case_mean_not_pooled(self):
        rows = [{"case_id": "a", "sample_id": "1", "value": 10},
                {"case_id": "b", "sample_id": "2", "value": 0},
                {"case_id": "b", "sample_id": "3", "value": 0}]
        summary, _, _ = summarize_rows(rows)
        self.assertEqual(summary["value"], 5)

    def test_factor_recalls(self):
        u = np.array([1., 2., -1., -2.]); a = np.ones(4); s = np.array([-1., 1., -1., -1.])
        m = factor_metrics(u, a, s)
        self.assertEqual(m["positive_recall"], .5); self.assertEqual(m["negative_recall"], 1.)

    def test_save_npz_is_non_overwriting(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"a.npz"; save_npz(p, {"x": np.arange(3)})
            with self.assertRaises(FileExistsError): save_npz(p, {"x": np.arange(5)})

    def test_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); io.write(p/"a.json", {"x": 1}); io.finish(p, "ok")
            r = Receipt(p, "ok"); r.file("a.json")
            io.write(p/"a.json", {"x": 2}, replace=True)
            with self.assertRaises(ValueError): r.file("a.json")

    def test_weight_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"weights.pt"; state = CoefficientHead(2).state_dict()
            save_weights(p, state)
            self.assertEqual(state_digest(state), state_digest(torch.load(p, weights_only=True)))
            with self.assertRaises(FileExistsError): save_weights(p, state)

    def test_small_head_can_reduce_product_loss(self):
        head = CoefficientHead(2); opt = torch.optim.Adam(head.parameters(), lr=.02)
        condition = torch.ones(1, 2, 4, 4, 4); magnitude = torch.ones(1, 4, 4, 4)
        target = torch.full_like(magnitude, .25)
        before = float(((torch.tanh(head(condition)/2)*magnitude-target)**2).mean().detach())
        for _ in range(16):
            opt.zero_grad(); loss = ((torch.tanh(head(condition)/2)*magnitude-target)**2).mean(); loss.backward(); opt.step()
        after = float(((torch.tanh(head(condition)/2)*magnitude-target)**2).mean().detach())
        self.assertLess(after, before)

    def test_frozen_envelope_bound(self):
        u = np.array([-3., -.2, 0., .5, 3.]); a = np.array([1., 1., 0., 1., 2.])
        s = np.divide(u, a, out=np.zeros_like(u), where=a>0).clip(-1, 1)
        np.testing.assert_allclose(abs(a*s-u), np.maximum(abs(u)-a, 0))

    def test_weighted_bernoulli_population_identity(self):
        u = np.array([-2., 1., 4.]); p = np.array([.3, .5, .2])
        magnitude = float(np.sum(p*np.abs(u)))
        positive = float(np.sum(p*np.abs(u)*(u>0))/magnitude)
        self.assertAlmostEqual(magnitude*(2*positive-1), float(np.sum(p*u)))

    def test_profile_gradient_is_finite(self):
        b = self.batch(); x = torch.randn_like(b["residual"], requires_grad=True)
        norm = {k: 1. for k in ("sign_bce", "reconstruction_mse", "profile_relative", "profile_absolute_mse")}
        loss, _ = objective(x, b, norm, self.cfg, "profile")
        grad = torch.autograd.grad(loss, x)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))

if __name__ == "__main__":
    unittest.main()
