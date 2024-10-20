"""Contains transients models."""

import copy
import abc
import numpy as np
import pandas as pd
import sncosmo as snc
from .constants import C_LIGHT_KMS
from . import utils as ut
from . import hosts as hst
from . import plasticc_model as plm


class AstrObj(abc.ABC):
    """Basic class for transients."""

    _type = ""
    _base_attrs = [
        "ID",
        "ra",
        "dec",
        "zcos",
        "vpec",
        "zpcmb",
        "como_dist",
        "model_name",
        "host_noise",
    ]

    _obj_attrs = [""]
    _available_models = [""]

    def __init__(self, sim_par, mag_fun=None, effects=None):
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
        mag_fun : str, optional
            The function used to compute the abs mag, by default None
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

        self._mag_fun = mag_fun

        # -- Set some default values
        if "ID" not in self._sim_par:
            self._sim_par["ID"] = 0
        if "host_noise" not in self._sim_par:
            self._sim_par["host_noise"] = False

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
        snc_source = self.source

        if "model_version" not in self._sim_par:
            self._sim_par["model_version"] = snc_source.version

        if effects is not None:
            eff_sources = [eff["source"] for eff in effects]
            eff_names = [eff["name"] for eff in effects]
            eff_frames = [eff["frame"] for eff in effects]
        else:
            eff_sources = None
            eff_names = None
            eff_frames = None

        model = snc.Model(
            source=snc_source,
            effects=eff_sources,
            effect_names=eff_names,
            effect_frames=eff_frames,
        )

        effect_par = {}
        for eff, eff_name in zip(model.effects, model.effect_names):
            for k in eff.param_names:
                eff_par = eff_name + k
                if eff_par in self._sim_par:
                    effect_par[eff_par] = self._sim_par[eff_par]

        model.set(t0=self._sim_par["t0"], z=self.zobs, **effect_par)

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
            (obs["time"] > self.sim_model.mintime())
            & (obs["time"] < self.sim_model.maxtime())
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

        sig_host = 0
        #compute the Noise from the host galaxy if required
        if self._sim_par["host_noise"]:
            sig_host = hst.model_host_noise(self._sim_par, obs)


        # -- Noise computation : Poisson Noise + Skynoise + ZP noise + Host gal Noise
        fluxerrtrue = np.sqrt(
            np.abs(fluxtrue) / obs['gain']
            + obs['skynoise']**2
            + (np.log(10) / 2.5 * fluxtrue * obs['sig_zp']) ** 2 
            + sig_host**2
        )

        gen = np.random.default_rng(random_seeds[1])
        flux = fluxtrue + gen.normal(loc=0.0, scale=fluxerrtrue)
        fluxerr = np.sqrt(fluxerrtrue**2 + (np.abs(flux) - np.abs(fluxtrue)) / obs['gain'])

        # -- Set magnitude
        mag = np.zeros_like(flux)
        magerr = np.zeros_like(flux)

        positive_fmask = flux > 0
        flux_pos = flux[positive_fmask]

        mag[positive_fmask] = -2.5 * np.log10(flux_pos) + obs["zp"][positive_fmask]

        magerr[positive_fmask] = (
            2.5 / np.log(10) * fluxerr[positive_fmask] / flux_pos
        )

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
        
        if self._sim_par["host_noise"]:
            sim_lc['host_noise'] = sig_host

        snc_par = {k: v for k, v in zip(self.sim_model.param_names, self.sim_model.parameters) if k!= 'z'}
        sim_lc.attrs = {
            "mu": self.mu,
            "zobs": self.zobs,
            "zCMB": self.zCMB,
            "effects": self.sim_model.effect_names,
            **snc_par,
            **self._sim_par
        }

        sim_lc.reset_index(inplace=True, drop=True)
        sim_lc.index.set_names("epochs", inplace=True)
        return sim_lc

    def mag_restframeband_to_amp(self, mag, band, magsys, amp_param_name='x0'):
        source = self.source
        m_current = source.peakmag(band, magsys)
        return 10.**(0.4 * (m_current - mag)) * source.get(amp_param_name)
        
    @property
    def source(self):
        if "model_version" not in self._sim_par:
            version = None
        else:
            version = self._sim_par["model_version"]

        return snc.get_source(name=self._sim_par["model_name"], version=version)
        
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
            5
            * np.log10(
                (1 + self.zcos)
                * (1 + self.zpcmb)
                * (1 + self.zpec) ** 2
                * self.como_dist
            )
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
            Raises if you use mag_fun 'salttripp' for a non salt model.
        ValueError
            Raises if you use a non-implemented mag_fun.
        ValueError
            Raises if you use mass-step without host logmass.
        """

        if self._mag_fun is None:
            self._mag_fun = "salttripp"

        if self._mag_fun.lower() == "salttripp":
            if model.source.name not in ["salt2", "salt3"]:
                raise ValueError("SALTTripp only available for salt2 & salt3 models")

            self._obj_attrs.extend(["alpha", "beta", "x0", "x1", "c"])

            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {coherent intrinsic scattering}
            mb = (
                self.SALTTripp(
                    self._sim_par["M0"],
                    self._sim_par["alpha"],
                    self._sim_par["beta"],
                    self._sim_par["x1"],
                    self._sim_par["c"],
                    self._sim_par["coh_sct"],
                )
                + self.mu
            )
        else:
            # TODO - BC : Find a way to use lambda function for mag_fun
            raise ValueError("mag_fun not available")
        
        # Add mass step
        self._obj_attrs.extend(['mass_step'])
        mb += self._sim_par["mass_step"]

        self._sim_par["mb"] = mb

        # Compute the x0 parameter
        self._sim_par["x0"] = self.mag_restframeband_to_amp(self._sim_par["mb"], 'bessellb', 'ab')
        
        # Set x1 and c
        model.set(x0=self._sim_par["x0"], x1=self._sim_par["x1"], c=self._sim_par["c"])
        return model

    @staticmethod
    def SALTTripp(M0, alpha, beta, x1, c, coh_sct):
        return M0 - alpha * x1 + beta * c + coh_sct


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


class SNIax(AstrObj):
    """SNiax class.

    Parameters
    ----------
    sn_par : dict
        Parameters of the object.

      | same as BasicAstrObj parameters
      | └── mag_sct, coherent mag scattering.
    sim_model : sncosmo.Model
        sncosmo Model to use.
    model_par : dict
        General model parameters.

      | same as BasicAstrObj model_par
      | ├── M0,  absolute magnitude
      | ├── sigM, sigma of coherent scattering
      | └── used model parameters
    """

    _obj_attrs = ["M0", "amplitude", "mb"]
    _type = "snIax"
    _available_models = plm.get_sed_listname("sniax")

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

        M0 = (
            model.source_peakmag("bessellv", "ab") + 0.345
        )  # correction to recalibrate to plasticc models
        self._sim_par["M0"] = M0

        m_v = self.mu + M0

        # Compute the amplitude  parameter
        model.set_source_peakmag(m_v, "bessellv", "ab")
        self._sim_par["mb"] = model.source_peakmag("bessellb", "ab")
        self._sim_par["amplitude"] = model.get("amplitude")

        dust = snc.CCM89Dust()
        model.add_effect(dust, frame="rest", name="host_")
        model.set(
            **{"host_ebv": self._sim_par["E_dust"], "host_r_v": self._sim_par["RV"]}
        )

        return model


class SNIa91bg(AstrObj):
    """SNia91bg class.

    Parameters
    ----------
    sn_par : dict
        Parameters of the object.

      | same as BasicAstrObj parameters
      | └── mag_sct, coherent mag scattering.
    sim_model : sncosmo.Model
        sncosmo Model to use.
    model_par : dict
        General model parameters.

      | same as BasicAstrObj model_par
      | ├── M0,  absolute magnitude
      | ├── sigM, sigma of coherent scattering
      | └── used model parameters
    """

    _obj_attrs = ["M0", "amplitude", "mb"]
    _type = "snIa91bg"
    _available_models = plm.get_sed_listname("snia91bg")

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

        M0 = model.source_peakmag("bessellv", "ab")
        self._sim_par["M0"] = M0

        m_v = self.mu + M0

        # Compute the amplitude  parameter
        model.set_source_peakmag(m_v, "bessellv", "ab")
        self._sim_par["mb"] = model.source_peakmag("bessellb", "ab")
        self._sim_par["amplitude"] = model.get("amplitude")
        return model
