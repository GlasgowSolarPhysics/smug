from copy import copy
import astropy.units as u
import numpy as np
import torch
from weno4 import weno4


class RadynversionAdapter:
    def __init__(
        self,
        model,
        atmos_params,
        line_profiles,
        line_half_width,
        z_stratification,
        dev=None,
    ):
        self.model = model
        self.atmos_params = atmos_params
        self.line_profiles = line_profiles
        self.line_half_width = line_half_width
        self.z_stratification = z_stratification
        self.dev = dev
        if dev is None:
            self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def to_tensor(x):
        return torch.from_numpy(x).float()

    @staticmethod
    def transform_ne(ne):
        return torch.log10(ne)

    @staticmethod
    def transform_temperature(temp):
        return torch.log10(temp)

    @staticmethod
    def transform_vel(v):
        vel_sign = v / torch.abs(v)
        vel_sign[torch.isnan(vel_sign)] = 0.0
        vel = vel_sign * torch.log10(torch.abs(v) + 1.0)
        return vel

    @staticmethod
    def strip_units_ne(ne):
        return (ne << u.cm ** (-3)).value

    @staticmethod
    def strip_units_temperature(vel):
        return (vel << u.K).value

    @staticmethod
    def strip_units_vel(vel):
        return (vel << (u.km / u.s)).value

    @staticmethod
    def inv_transform_ne(ne):
        return 10**ne

    @staticmethod
    def inv_transform_temperature(temp):
        return 10**temp

    @staticmethod
    def inv_transform_vel(v):
        v_sign = v / torch.abs(v)
        v_sign[torch.isnan(v_sign)] = 0.0
        vel = v_sign * (10 ** torch.abs(v) - 1.0)
        return vel

    def transform_atmosphere(self, /, **kwargs):
        missing_params = []
        for param in self.atmos_params:
            if not param in kwargs:
                missing_params.append(param)
        if len(missing_params) != 0:
            raise ValueError(f"Parameters missing: {missing_params}.")

        for param in self.atmos_params:
            shape = kwargs[param].shape
            if len(shape) != 2:
                kwargs[param] = kwargs[param][None, :]

        Nspace = self.model.atmos_size
        batch_size = kwargs[self.atmos_params[0]].shape[0]
        for param in self.atmos_params:
            shape = kwargs[param].shape
            if shape[0] != batch_size:
                raise ValueError(
                    f"Got batch_size of {shape[0]}, expected {batch_size} for parameter '{param}'."
                )
            if shape[1] != Nspace:
                raise ValueError(
                    f"Got atmos_size of {shape[1]}, expected {Nspace} for parameter '{param}'."
                )

        tensor_params = {
            param: self.to_tensor(getattr(self, f"strip_units_{param}")(kwargs[param]))
            for param in self.atmos_params
        }

        result = torch.zeros((batch_size, self.model.size))
        for i, param in enumerate(self.atmos_params):
            trans = getattr(self, f"transform_{param}")
            result[:, i * Nspace : (i + 1) * Nspace] = trans(tensor_params[param])

        return result

    def inv_transform_atmosphere(self, atmos):
        Nspace = self.model.atmos_size
        result = {}
        for i, param in enumerate(self.atmos_params):
            trans = getattr(self, f"inv_transform_{param}")
            result[param] = trans(atmos[:, i * Nspace : (i + 1) * Nspace])
        return result

    def line_grids(self):
        return {
            line: np.linspace(
                -self.line_half_width[i],
                self.line_half_width[i],
                self.model.line_profile_size,
            )
            for i, line in enumerate(self.line_profiles)
        }

    def interpolate_lines(self, lines, delta_lambdas):
        grids = self.line_grids()
        lines_in = copy(lines)
        for line, data in lines_in.items():
            if len(data.shape) == 1:
                lines_in[line] = data[None, :]

        # NOTE(cmo): Interpolate lines to grids
        interp_lines = {
            line: torch.zeros(array.shape[0], self.model.line_profile_size)
            for line, array in lines_in.items()
        }
        for line in interp_lines:
            result = interp_lines[line].numpy()
            for i in range(result.shape[0]):
                result[i, :] = weno4(
                    grids[line], delta_lambdas[line], lines_in[line][i]
                )
        return interp_lines

    def transform_lines(self, lines, delta_lambdas):
        interp_lines = self.interpolate_lines(lines, delta_lambdas)
        maxs = {
            line: torch.max(array, axis=1)[0] for line, array in interp_lines.items()
        }
        max_per_obs = torch.zeros(interp_lines[self.line_profiles[0]].shape[0])
        for _, max_val in maxs.items():
            max_per_obs = torch.maximum(max_val, max_per_obs)
        transformed_lines = {
            line: array / max_per_obs[:, None] for line, array in interp_lines.items()
        }
        return transformed_lines

    def forward_model(self, atmos, cpu=True):
        inp = atmos.to(self.dev)
        with torch.no_grad():
            out = self.model(inp)[0]

        result = {}
        result["LatentSpace"] = out[:, : self.model.latent_size]

        line_size = self.model.line_profile_size
        Nlines = len(self.line_profiles)
        for idx, line in enumerate(self.line_profiles):
            start = -(Nlines - idx) * line_size
            end = start + line_size
            if end == 0:
                end = None
            result[line] = out[:, start:end]
        if cpu:
            result = {k: v.cpu() for k, v in result.items()}

        return result

    def invert_lines(self, lines, latent_space=None, batch_size=None, cpu=True):
        if batch_size is None:
            batch_size = lines[self.line_profiles[0]].shape[0]

        if latent_space is None:
            latent_space = torch.randn(batch_size, self.model.latent_size)

        input = torch.zeros(batch_size, self.model.size)
        input[:, : self.model.latent_size] = latent_space
        line_size = self.model.line_profile_size
        Nlines = len(self.line_profiles)
        for idx, line in enumerate(self.line_profiles):
            start = -(Nlines - idx) * line_size
            end = start + line_size
            if end == 0:
                end = None
            input[:, start:end] = lines[line]

        input = input.to(self.dev)
        with torch.no_grad():
            result = self.model(input, rev=True)[0]
        if cpu:
            result = result.cpu()

        return result
