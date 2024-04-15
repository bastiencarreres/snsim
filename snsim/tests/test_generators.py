import snsim
import sncosmo as snc
import numpy as np
import pandas as pd
from numpy.testing import (
    assert_allclose,
    assert_approx_equal,
    assert_array_almost_equal,
)
from snsim import generators as gen


FlatSource = snc.TimeSeriesSource(
    np.linspace(0.0, 100.0, 10),
    np.linspace(800.0, 20000.0, 100),
    np.ones((10, 100), dtype=float),
)

snc.register(FlatSource, name="flatsource")


class FakeGen(gen.BaseGen):
    _object_type = "TimeSeries"
    _available_models = ["flatsource"]
    _available_rates = {"testrate": "lambda z: 1e-5 * z * ({h}/0.70)**3"}

    def gen_par(self, n_obj, basic_par, seed=None):
        return {}


class TestGenerators:
    def setup_class(self):
        self.config = {"M0": -19.0, "model_name": "flatsource"}

        # Set the cosmology (astropy.cosmology object)
        cosmology = {"name": "planck18"}
        self.cosmo = snsim.utils.set_cosmo(cosmology)

        # distribution of peculiar velocities of SNe
        self.vpec_dist = {"mean_vpec": 0, "sig_vpec": 0}

        self.time_range = [-1000, 1000]
        self.z_range = [0.001, 0.1]

    def test_rate(self):
        Gen_str_rate = FakeGen(
            {**self.config, "rate": "lambda z: 1e-5 * z"},
            self.cosmo,
            self.time_range,
            z_range=self.z_range,
            vpec_dist=self.vpec_dist,
        )

        rate = lambda z: 1e-5 * z
        Gen_lambda_rate = FakeGen(
            {**self.config, "rate": rate},
            self.cosmo,
            self.time_range,
            z_range=self.z_range,
            vpec_dist=self.vpec_dist,
        )

        Gen_register_rate = FakeGen(
            {**self.config, "rate": "testrate"},
            self.cosmo,
            self.time_range,
            z_range=self.z_range,
            vpec_dist=self.vpec_dist,
        )

        assert_approx_equal(Gen_str_rate.rate(2), 1e-5 * 2)
        (Gen_lambda_rate.rate(2), 1e-5 * 2)
        assert_approx_equal(
            Gen_register_rate.rate(2), 1e-5 * 2 * (self.cosmo.h / 0.70) ** 3
        )

    def test_zcdf(self):
        Gen_str_rate = FakeGen(
            {**self.config, "rate": "lambda z: 1e-5"},
            self.cosmo,
            self.time_range,
            z_range=self.z_range,
            vpec_dist=self.vpec_dist,
        )

        test_array = np.array(
            [
                3.45704207e-03,
                4.10371336e-01,
                1.46706705e00,
                3.13588799e00,
                5.38069438e00,
                8.16680786e00,
                1.14609591e01,
                1.52312376e01,
                1.94470438e01,
                2.40790431e01,
            ]
        )

        assert_array_almost_equal(Gen_str_rate._z_dist.pdfx[::1000], test_array)
