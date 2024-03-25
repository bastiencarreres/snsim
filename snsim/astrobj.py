"""Contains transients models."""

import copy
import abc
import numpy as np
import pandas as pd
import sncosmo as snc
from .constants import C_LIGHT_KMS
from . import utils as ut


class AstrObj(abc.ABC):
    """Basic class for transients."""

    _type = ""
    _base_attrs = ["ID", "ra", "dec", "zcos", "vpec", "zpcmb", "como_dist", "model_name"]

    _obj_attrs = [""]
    _available_models = [""]

    def __init__(self, sim_par, relation=None, effects=None):
        """Init AstrObj class.

        Parameters
        ----------
        sim_par: dict
            Simulation parameters.

            | sim_par
            | ├── zcos, cosmological redshift
            | ├── zpcmb, CMB dipole redshift contribution
            | ├── como_dist, comoving distance of the obj
            | ├── vpec, obj peculiar velocity
            | ├── ra, obj Right Ascension
            | ├── dec, obj Declinaison
            | ├── t0, obj peak time
            | └── sncosmo par
        relation : str, optional
            _description_, by default None
        effects : list, optional
            sncosmo effects dic, by default None

            | effect
            | ├── source: sncosmo.effect, sncosmo effect obj
            | ├── frame: str 'obs' or 'rest'
            | └── name: str, effect name

        Raises
        ------
        ValueError
            If simpar['model_name'] is not available
        """
        # -- Copy input parameters dic
        self._sim_par = copy.copy(sim_par)

        self._relation = relation

        if "ID" not in self._sim_par:
            self._sim_par["ID"] = 0

        # -- Check model name
        if self.sim_par["model_name"] not in self._available_models:
            raise ValueError(f"{self.sim_par['model_name']} not available.")

        # -- Update attrs
        for k in self._base_attrs:
            setattr(self, k, self._sim_par[k])

        # -- sncosmo model
        self._sim_model = self._init_model(effects)

        # -- Update attr of astrobj class
        for k in [*self._obj_attrs, "model_version"]:
            setattr(self, k, self._sim_par[k])

    @abc.abstractmethod
    def _set_model_par(self, model):
        """This method set model parameters that are not t0 or z."""
        pass

    def _init_model(self, effects):
        """Initialise sncosmo model using the good source.

        Parameters
        ----------
        effects : list, optional
            sncosmo effects dic, by default None

            | effect
            | ├── source: sncosmo.effect, sncosmo effect obj
            | ├── frame: str 'obs' or 'rest'
            | └── name: str, effect name

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.

        """
        if "model_version" not in self._sim_par:
            version = None
        else:
            version = self._sim_par["model_version"]

        snc_source = snc.get_source(name=self._sim_par["model_name"], version=version)

        if "model_version" not in self._sim_par:
            self._sim_par["model_version"] = snc_source.version

        if effects is not None:
            eff = [eff["source"] for eff in effects]
            eff_names = [eff["name"] for eff in effects]
            eff_frames = [eff["frame"] for eff in effects]
        else:
            eff = None
            eff_names = None
            eff_frames = None

        model = snc.Model(
            source=snc_source,
            effects=eff,
            effect_names=eff_names,
            effect_frames=eff_frames,
        )

        for eff in effects:
            for k in eff["source"].param_names:
                if k in self._sim_par:
                    model.set(self._sim_par[eff["name"] + k])

        model.set(t0=self._sim_par["t0"], z=self.zobs)

        model = self._set_model_par(model)

        return model

    def gen_flux(self, obs, mod_fcov=False, seed=None):
        """
        Generate the flux for given obs.

        Parameters
        ----------
        obs : pd.DataFrame
            Observations info.
        mod_fcov : bool, optional
            Either or not using model scattering, by default False
        seed : numpy.random.SeedSequence, optional
            numpy random seed, by default None

        Returns
        -------
        pd.DataFrane
            Flux of the AstrObj for given obs.

        Raises
        ------
        ValueError
            Raises if mod_fcov is not available for the used model
        """
        random_seeds = ut.gen_rndchilds(seed, size=2)

        # -- Check for fcov
        if mod_fcov:
            if not hasattr(self.sim_model, "bandfluxcov"):
                raise ValueError("This sncosmo model has no flux covariance available")

        # mask to delete observed points outside time range of the model
        obs = obs[
            (obs["time"] > self.sim_model.mintime()) & (obs["time"] < self.sim_model.maxtime())
        ]

        if mod_fcov:
            # -- Implement the flux variation due to simulation model covariance
            gen = np.random.default_rng(random_seeds[0])
            fluxtrue, fluxcov = self.sim_model.bandfluxcov(
                obs["band"], obs["time"], zp=obs["zp"], zpsys=obs["zpsys"]
            )

            fluxtrue += gen.multivariate_normal(
                np.zeros(len(fluxcov)), fluxcov, check_valid="ignore", method="eigh"
            )

        else:
            fluxtrue = self.sim_model.bandflux(
                obs["band"], obs["time"], zp=obs["zp"], zpsys=obs["zpsys"]
            )

        # -- Noise computation : Poisson Noise + Skynoise + ZP noise
        fluxerrtrue = np.sqrt(
            np.abs(fluxtrue) / obs.gain
            + obs.skynoise**2
            + (np.log(10) / 2.5 * fluxtrue * obs.sig_zp) ** 2
        )

        gen = np.random.default_rng(random_seeds[1])
        flux = fluxtrue + gen.normal(loc=0.0, scale=fluxerrtrue)
        fluxerr = np.sqrt(
            np.abs(flux) / obs.gain + obs.skynoise**2 + (np.log(10) / 2.5 * flux * obs.sig_zp) ** 2
        )

        # -- Set magnitude
        mag = np.zeros_like(flux)
        magerr = np.zeros_like(flux)

        positive_fmask = flux > 0
        flux_pos = flux[positive_fmask]

        mag[positive_fmask] = -2.5 * np.log10(flux_pos) + obs["zp"][positive_fmask]

        magerr[positive_fmask] = 2.5 / np.log(10) * 1 / flux_pos * fluxerr[positive_fmask]

        mag[~positive_fmask] = np.nan
        magerr[~positive_fmask] = np.nan

        # -- Create DataFrame of the lightcurve
        sim_lc = pd.DataFrame(
            {
                "time": obs["time"],
                "fluxtrue": fluxtrue,
                "fluxerrtrue": fluxerrtrue,
                "flux": flux,
                "fluxerr": fluxerr,
                "mag": mag,
                "magerr": magerr,
                "zp": obs["zp"],
                "zpsys": obs["zpsys"],
                "gain": obs["gain"],
                "skynoise": obs["skynoise"],
            }
        )

        # TODO - BC: Maybe remove this "for loop"
        for k in obs.columns:
            if k not in sim_lc.columns:
                sim_lc[k] = obs[k].values

        sim_lc.attrs = {"mu": self.mu, "zobs": self.zobs, "zCMB": self.zCMB, **self._sim_par}

        sim_lc.reset_index(inplace=True, drop=True)
        sim_lc.index.set_names("epochs", inplace=True)
        return sim_lc

    @property
    def zpec(self):
        """Get peculiar velocity redshift."""
        return self.vpec / C_LIGHT_KMS

    @property
    def zCMB(self):
        """Get CMB frame redshift."""
        return (1 + self.zcos) * (1 + self.zpec) - 1.0

    @property
    def zobs(self):
        """Get observed redshift."""
        return (1 + self.zcos) * (1 + self.zpec) * (1 + self.zpcmb) - 1.0

    @property
    def mu(self):
        """Get distance moduli."""
        return (
            5 * np.log10((1 + self.zcos) * (1 + self.zpcmb) * (1 + self.zpec) ** 2 * self.como_dist)
            + 25
        )

    @property
    def sim_model(self):
        return self._sim_model

    @property
    def sim_par(self):
        return self._sim_par


class SNIa(AstrObj):
    """SNIa class. Inherit from AstrObj."""

    _type = "snIa"
    _available_models = ["salt2", "salt3"]
    _obj_attrs = ["M0", "mb", "coh_sct"]

    def _set_model_par(self, model):
        """
        Set sncosmo model parameters.

        Parameters
        ----------
        model : sncosmo.Model
            The sncosmo model.

        Returns
        -------
        sncosmo.Model
            The sncosmo model with parameters set.

        Raises
        ------
        ValueError
            Raises if you use relation 'salttripp' for a non salt model.
        ValueError
            Raises if you use a non-implemented relation.
        ValueError
            Raises if you use mass-step without host logmass.
        """
        M0 = self._sim_par["M0"] + self._sim_par["coh_sct"]

        if self._relation is None:
            self._relation = "salttripp"

        if self._relation.lower() == "salttripp":
            if model.source.name not in ["salt2", "salt3"]:
                raise ValueError("SALTTripp only available for salt2 & salt3 models")

            self._obj_attrs.extend(["alpha", "beta", "x0", "x1", "c"])

            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {coherent intrinsic scattering}
            self._sim_par["mb"] = (
                self.SALTTripp(
                    M0,
                    self._sim_par["alpha"],
                    self._sim_par["beta"],
                    self._sim_par["x1"],
                    self._sim_par["c"],
                )
                + self.mu
            )

        else:
            # TODO - BC : Find a way to use lambda function for relation
            raise ValueError("Relation not available")

        if "mass_step" in self._sim_par:
            if "host_mass" in self._sim_par:
                if self._sim_par["host_mass"] > 10.0:
                    mb += self._sim_par["mass_step"]
            else:
                raise ValueError("Provide SN host mass to account for the magnitude mass step")

        # Set x1 and c
        model.set(x1=self._sim_par["x1"], c=self._sim_par["c"])

        # Compute the x0 parameter
        model.set_source_peakmag(self._sim_par["mb"], "bessellb", "ab")
        self._sim_par["x0"] = model.get("x0")
        return model

    @staticmethod
    def SALTTripp(M0, alpha, beta, x1, c):
        return M0 - alpha * x1 + beta * c


class TimeSeries(AstrObj):
    """TimeSeries class."""

    _obj_attrs = ["M0", "amplitude", "mb", "coh_sct"]

    def _set_model_par(self, model):
        """Set sncosmo model parameters.

        Parameters
        ----------
        model : sncosmo.Model
            The sncosmo model.

        Returns
        -------
        sncosmo.Model
            The sncosmo model with parameters set.
        """

        M0 = self._sim_par["M0"] + self._sim_par["coh_sct"]

        m_r = self.mu + M0

        # Compute the amplitude  parameter
        model.set_source_peakmag(m_r, "bessellr", "ab")
        self._sim_par["mb"] = model.source_peakmag("bessellb", "ab")
        self._sim_par["amplitude"] = model.get("amplitude")
        return model


class SNII(TimeSeries):
    """SNII class. Inherit from TimeSeries."""

    _type = "snII"
    _available_models = ut.Templatelist_fromsncosmo("snii")


class SNIIpl(TimeSeries):
    """SNII P/L class. Inherit from TimeSeries."""

    _type = "snIIpl"
    _available_models = ut.Templatelist_fromsncosmo("sniipl")


class SNIIb(TimeSeries):
    """SNIIb class. Inherit from TimeSeries."""

    _type = "snIIb"
    _available_models = ut.Templatelist_fromsncosmo("sniib")


class SNIIn(TimeSeries):
    """SNIIn class. Inherit from TimeSeries."""

    _type = "snIIn"
    _available_models = ut.Templatelist_fromsncosmo("sniin")


class SNIbc(TimeSeries):
    """SNIb/c class. Inherit from TimeSeries."""

    _type = "snIb/c"
    _available_models = ut.Templatelist_fromsncosmo("snib/c")


class SNIc(TimeSeries):
    """SNIIn class. Inherit from TimeSeries."""

    _type = "snIc"
    _available_models = ut.Templatelist_fromsncosmo("snic")


class SNIb(TimeSeries):
    """SNIIn class. Inherit from TimeSeries."""

    _type = "snIb"
    _available_models = ut.Templatelist_fromsncosmo("snib")


class SNIc_BL(TimeSeries):
    """SNIIn class. Inherit from TimeSeries."""

    _type = "snIc-BL"
    _available_models = ut.Templatelist_fromsncosmo("snic-bl")
