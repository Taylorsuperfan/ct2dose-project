import unittest
import numpy as np
import torch
from cwfr.model import ConditionalWFR
from cwfr.common import seed_all
from cwfr.metrics import score,aggregate

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):seed_all(17)
    def test_initial_zero_correction(self):
        m=ConditionalWFR(2,8);c=torch.randn(1,2,8,8,8);z=m.predict(c,2,256)
        np.testing.assert_allclose(z['residual_normalized'],0);self.assertAlmostEqual(z['magnitude'].sum(),512)
    def test_target_free_signature(self):
        import inspect
        self.assertEqual(list(inspect.signature(ConditionalWFR.predict).parameters),['self','condition','steps','query_chunk'])
    def test_field_shapes_and_gradients(self):
        m=ConditionalWFR(2,8);c=torch.randn(2,2,8,8,8);x=torch.rand(2,11,3);t=torch.rand(2,11)
        v,g,r=m.vector_field(x,t,m.encode_transport(c));self.assertEqual(v.shape,(2,11,3));self.assertEqual(g.shape,(2,11))
        (v.sum()+g.sum()+m.sign_logits(c).sum()).backward()
        self.assertIsNotNone(m.field[-1].weight.grad);self.assertIsNotNone(m.sign_out.weight.grad)
    def test_condition_not_modified(self):
        m=ConditionalWFR(2,8);c=torch.randn(1,2,8,8,8);old=c.clone();m.predict(c,2,100)
        torch.testing.assert_close(c,old)
    def test_query_chunk_invariance(self):
        m=ConditionalWFR(2,8);torch.nn.init.normal_(m.field[-1].weight,std=.01);c=torch.randn(1,2,8,8,8)
        a=m.predict(c,2,512);b=m.predict(c,2,100)
        np.testing.assert_allclose(a['magnitude'],b['magnitude'],atol=1e-6,rtol=1e-6)
    def test_growth_scales_weights(self):
        m=ConditionalWFR(2,8)
        with torch.no_grad():m.field[-1].bias[3]=np.log(2)
        z=m.predict(torch.zeros(1,2,4,4,4),4,64)
        self.assertAlmostEqual(z['diagnostics']['predicted_magnitude_total'],128,places=5)
    def test_soft_sign_bound(self):
        m=ConditionalWFR(2,8)
        with torch.no_grad():m.sign_out.bias.fill_(-2)
        z=m.predict(torch.zeros(1,2,4,4,4),2,64)
        self.assertTrue((z['residual_normalized']<0).all());self.assertLess(z['diagnostics']['weighted_absolute_sign'],1)
    def test_global_metrics(self):
        y=np.ones((4,4,4));p=y+.2;r=score(p,y,1000)
        self.assertAlmostEqual(r['rmse_stored_units'],.2);self.assertAlmostEqual(r['mae_stored_units'],.2)
    def test_axis_mapping(self):
        y=np.zeros((3,4,5));y[1,2,3]=1;p=y.copy();p[1,2,0]=.5
        r=score(p,y,1);self.assertGreater(r['x_profile_rmse_stored_units'],0)
        self.assertEqual(r['y_profile_rmse_stored_units'],0);self.assertEqual(r['z_profile_rmse_stored_units'],0)
    def test_legacy_threshold_and_epsilon(self):
        y=np.ones((3,3,3))*.001;p=y*1.1;r=score(p,y,1000)
        self.assertAlmostEqual(r['x_mean_pct'],100*.1/(1+1e-8),places=10)
    def test_equal_case_aggregation(self):
        rows=[dict(method='m',sample_id=str(i),case_id='a',rmse_stored_units=1.) for i in range(3)]
        rows+=[dict(method='m',sample_id='x',case_id='b',rmse_stored_units=3.)]
        _,_,s=aggregate(rows);self.assertEqual(s.rmse_stored_units.iloc[0],2)
    def test_duplicate_metrics_rejected(self):
        a=dict(method='m',sample_id='a',case_id='x',rmse_stored_units=1)
        with self.assertRaises(ValueError):aggregate([a,a])
