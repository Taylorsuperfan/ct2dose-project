import unittest
import numpy as np
from dataclasses import replace
from cwfr.geometry import centers,cost_matrix,mm_uot,make_bank,sample_pairs,path_targets,deposit,objective
from cwfr.config import Config

class MathTests(unittest.TestCase):
    def test_single_point_oet(self):
        G,info,g0,g1=mm_uot([1.],[4.],np.zeros((1,1)))
        self.assertAlmostEqual(G.item(),2.,places=12);self.assertEqual(g0.item(),1.);self.assertEqual(g1.item(),4.)
    def test_one_point_distance(self):
        C=np.array([[.6]]);G,*_=mm_uot([2.],[3.],C)
        self.assertAlmostEqual(G.item(),np.sqrt(6)*np.exp(-.3),places=12)
    def test_general_semicoupling(self):
        a=np.array([.2,.3,.5]);b=np.array([.8,.1]);C=np.array([[0.,.4],[.3,.1],[.6,.2]])
        G,stats,g0,g1=mm_uot(a,b,C,500)
        np.testing.assert_allclose(g0.sum(1),a);np.testing.assert_allclose(g1.sum(0),b)
        self.assertLessEqual(stats['final_objective'],stats['initial_objective'])
    def test_solver_against_convex_numeric_reference(self):
        from scipy.optimize import minimize
        a=np.array([.4,.6]);b=np.array([.5,.3]);C=np.array([[.1,.8],[.9,.2]])
        G,stats,*_=mm_uot(a,b,C,3000,1e-12)
        f=lambda z:objective(z.reshape(2,2),a,b,C)
        ref=minimize(f,np.outer(a,b).ravel(),bounds=[(1e-15,None)]*4,method='L-BFGS-B',options={'ftol':1e-13,'maxiter':5000})
        self.assertLess(abs(f(G.ravel())-ref.fun),1e-7)
    def test_solver_rejects_zero_weights(self):
        with self.assertRaises(ValueError):mm_uot([0],[1],np.zeros((1,1)))
    def test_cost_symmetry(self):
        x=centers((2,2,2));np.testing.assert_allclose(cost_matrix(x,x,1),cost_matrix(x,x,1).T)
    def test_cost_cutoff(self):
        with self.assertRaises(ValueError):cost_matrix(np.zeros((1,3)),np.full((1,3),10.),1)
    def test_path_endpoints(self):
        x0=np.array([[.1,.2,.3],[.7,.4,.2]]);x1=np.array([[.8,.6,.5],[.7,.4,.2]]);mass=np.array([3.,.2])
        p,_,_,m=path_targets(x0,x1,mass,[0,0],1);np.testing.assert_allclose(p,x0);np.testing.assert_allclose(m,1)
        p,_,_,m=path_targets(x0,x1,mass,[1,1],1);np.testing.assert_allclose(p,x1);np.testing.assert_allclose(m,mass)
    def test_same_location_midpoint(self):
        x=np.ones((1,3))*.4;p,v,g,m=path_targets(x,x,[4],[.5],1)
        self.assertAlmostEqual(m.item(),2.25);np.testing.assert_allclose(v,0);self.assertAlmostEqual(g.item(),4/3)
    def test_path_derivative(self):
        x=np.array([[.2,.3,.4]]);y=np.array([[.7,.6,.5]]);t=.38;h=1e-6
        p,v,g,m=path_targets(x,y,[2.3],[t],1)
        pp,_,_,mp=path_targets(x,y,[2.3],[t+h],1);pm,_,_,mm=path_targets(x,y,[2.3],[t-h],1)
        np.testing.assert_allclose((pp-pm)/(2*h),v,atol=1e-9)
        np.testing.assert_allclose((mp-mm)/(2*h)/m,g,atol=1e-9)
    def test_bank_preserves_total_ratio(self):
        cfg=Config(source_points_per_bank=8,target_draws_per_bank=12,banks_per_record=1,coupling_iterations=100)
        bank,stats=make_bank(np.ones((4,4,4))*2,cfg,np.random.default_rng(1))
        q=np.diff(np.r_[0,bank['cdf']]).reshape(len(bank['x0']),len(bank['x1']))
        self.assertAlmostEqual(np.sum(q*bank['row_ratio'][:,None]*bank['col_ratio'][None,:]),2.,places=10)
    def test_target_mass_not_normalized_away(self):
        cfg=Config(source_points_per_bank=8,target_draws_per_bank=8,coupling_iterations=30)
        _,s=make_bank(np.ones((4,4,4))*3,cfg,np.random.default_rng(3));self.assertEqual(s['target_total'],3.)
    def test_zero_target_stops(self):
        with self.assertRaises(ValueError):make_bank(np.zeros((4,4,4)),Config(),np.random.default_rng())
    def test_sample_is_reproducible(self):
        cfg=Config(source_points_per_bank=8,target_draws_per_bank=8,coupling_iterations=20)
        bank,_=make_bank(np.ones((4,4,4)),cfg,np.random.default_rng(1))
        left=sample_pairs(bank,100,np.random.default_rng(17));right=sample_pairs(bank,100,np.random.default_rng(17))
        for a,b in zip(left,right):np.testing.assert_array_equal(a,b)
    def test_center_roundtrip(self):
        shape=(4,5,6);w=np.arange(np.prod(shape))+1.;x=centers(shape)
        np.testing.assert_allclose(deposit(x,w,shape),w.reshape(shape),atol=1e-12)
    def test_boundary_total(self):
        p=np.array([[0,0,0],[1,1,1],[.123,.245,.384]])
        self.assertAlmostEqual(deposit(p,[1,2,3],(4,4,4)).sum(),6)
    def test_signed_net_conservation(self):
        rng=np.random.default_rng(1);w=rng.normal(size=100)
        self.assertAlmostEqual(deposit(rng.random((100,3)),w,(4,4,4)).sum(),w.sum(),places=12)
    def test_outside_stops(self):
        with self.assertRaises(ValueError):deposit(np.array([[-.01,.5,.5]]),[1],(4,4,4))
    def test_grid_sign_order(self):
        p=np.array([[.5,.5,.5]]);mag=deposit(p,[1],(2,2,2));sign=np.ones((2,2,2));sign[0]=-1
        self.assertNotAlmostEqual((mag*sign).sum(),deposit(p,[1],(2,2,2)).sum())
    def test_config_cutoff(self):
        with self.assertRaises(ValueError):Config(delta=.1).validate()
    def test_config_matrix_limit(self):
        with self.assertRaises(ValueError):Config(source_points_per_bank=32768,target_draws_per_bank=32768).validate()
