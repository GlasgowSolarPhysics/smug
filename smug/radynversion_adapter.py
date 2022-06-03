from copy import copy

import astropy.units as u
import numpy as np
import torch
from astropy.stats import median_absolute_deviation as mad
from crispy import CRISPSequence, Inversion, ObjDict
from tqdm import tqdm
from weno4 import weno4


class RadynversionAdapter:
    """
    Run a RADYNVERSION model: prepare the data, run the model, and extract the
    output.

    The parameters for standard uses of this class are present in the
    `smug.radynversion_model.model_params` dict.  When passing multiple
    instances of a parameter to any of the methods of this class (unless
    otherwise specified) they are expected to have leading dimension the number
    of instances, and second dimension the size of the parameter.

    Transformation of different parameters is controlled by the
    `transform_{param}` and `inv_transform_{param}` static methods on the class,
    and these are looked up dynamically, i.e. this can be used with updated
    models taking different parameters by subclassing/monkeypatching.

    Parameters
    ----------
    model : RadynversionModel
        The model to use.
    atmos_params : list of str
        The names of the atmospheric parameters in the order expected by the
        model.
    line_profiles : list of str
        The names of the spectral lines in the order expected by the model.
    line_half_width : list of float
        The half width (in Angstrom) of the lines expected by the model.
    z_stratification : array-like
        The z_stratification used in the model.
    dev : `torch.device`, optional
        The device to run the model on. Default will use the first CUDA device
        (if available), or CPU.
    """
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
            self.model.to(self.dev)
        self.model.eval()

    @staticmethod
    def to_tensor(x):
        """Convert a numpy array to torch Tensor and ensure float32 dtype."""
        return torch.from_numpy(x).float()

    @staticmethod
    def transform_ne(ne):
        """Transform ne for input into the network (log10)."""
        return torch.log10(ne)

    @staticmethod
    def transform_temperature(temp):
        """Transform temperature for input into the network (log10)."""
        return torch.log10(temp)

    @staticmethod
    def transform_vel(v):
        """Transform velocity for input into the network."""
        vel_sign = v / torch.abs(v)
        vel_sign[torch.isnan(vel_sign)] = 0.0
        vel = vel_sign * torch.log10(torch.abs(v) + 1.0)
        return vel

    @staticmethod
    def strip_units_ne(ne):
        """Convert ne to astropy units of cm^{-3} and return the value."""
        return (ne << u.cm ** (-3)).value

    @staticmethod
    def strip_units_temperature(vel):
        """Convert temperature to astropy units of K and return the value."""
        return (vel << u.K).value

    @staticmethod
    def strip_units_vel(vel):
        """Convert temperature to astropy units of km/s and return the value."""
        return (vel << (u.km / u.s)).value

    @staticmethod
    def inv_transform_ne(ne):
        """Inverse transform ne for output from the network."""
        return 10**ne

    @staticmethod
    def inv_transform_temperature(temp):
        """Inverse transform temperature for output from the network."""
        return 10**temp

    @staticmethod
    def inv_transform_vel(v):
        """Inverse transform velocity for output from the network."""
        v_sign = v / torch.abs(v)
        v_sign[torch.isnan(v_sign)] = 0.0
        vel = v_sign * (10 ** torch.abs(v) - 1.0)
        return vel

    def transform_atmosphere(self, /, **kwargs):
        """Construct a tensor of transformed atmospheric parameters for input
        into the network.

        Default units (if not using astropy units): ne: cm^{-3}, temperature: K,
        velocity: km/s

        Parameters
        ----------
        **kwargs
            key, value mapping of the atmospheric parameters. Optionally using
            astropy units which will be converted.

        Returns
        -------
        atmosphere: torch.Tensor
            The atmospheric input prepared for the network.
        """
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

    def inv_transform_atmosphere(self, atmos, ignore=None):
        """Convert and unpack a transformed atmosphere into a dict by parameter.

        Parameters
        ----------
        atmos : array-like
            The packed atmosphere to expand
        ignore : list of str
            Parameters to ignore in the inverse transformation step.

        Returns
        -------
        dict of array-like
            The atmospheric parameters
        """
        Nspace = self.model.atmos_size
        result = {}
        for i, param in enumerate(self.atmos_params):
            if ignore is not None and param in ignore:
                result[param] = atmos[:, i * Nspace : (i + 1) * Nspace]
            else:
                trans = getattr(self, f"inv_transform_{param}")
                result[param] = trans(atmos[:, i * Nspace : (i + 1) * Nspace])
        return result

    def line_grids(self):
        """The wavelength grid expected for each line, in a dict by line name.
        Wavelengths are in Angstrom relative to rest wavelength.
        """
        return {
            line: np.linspace(
                -self.line_half_width[i],
                self.line_half_width[i],
                self.model.line_profile_size,
            )
            for i, line in enumerate(self.line_profiles)
        }

    def interpolate_lines(self, lines, delta_lambdas):
        """Interpolate lines from their current grid to the one expected by the
        model.

        Parameters
        ----------
        lines : dict of str to array-like
            The lines to interpolate, with their expected name as key (as
            detailed in self.line_profiles)
        delta_lambdas : dict of str to array-like
            The wavelength grids for the lines (in Angstrom), relative to the
            rest wavelength.

        Returns
        -------
        dict of str to array-like
            The interpolated lines on the grids given by `self.line_grids()`.
        """
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
        """Interpolate and transform (normalise) lines ready for use in the
        network.

        Parameters
        ----------
        lines : dict of str to array-like
            The lines to interpolate, with their expected name as key (as
            detailed in self.line_profiles)
        delta_lambdas : dict of str to array-like
            The wavelength grids for the lines (in Angstrom), relative to the
            rest wavelength.

        Returns
        -------
        dict of str to array-like
            The transformed lines on the grids given by `self.line_grids()`.
        """
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
        """
        Compute the line profiles associated with atmospheric samples and return
        in a dict by line name. This result also contains the associated latent
        space under key `LatentSpace`.

        Parameters
        ----------
        atmos : torch.Tensor
            The pre-packed atmosphere tensor (see `transform_atmosphere`).
        cpu : bool, optional
            Whether to return the line profile tensors to the cpu. Default: True.

        Returns
        -------
        dict of str to tensor
            The line profiles associated with these atmospheres.
        """
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

    def line_slice(self, idx):
        """The slice associated with line of index `idx` in the packed
        [latent_space, lines] vector for the reverse process."""
        num_lines = len(self.line_profiles)
        line_size = self.model.line_profile_size
        line_start = -(num_lines - idx) * line_size
        line_end = line_start + line_size
        if line_end == 0:
            line_end = None
        return slice(line_start, line_end)

    def invert_lines(
        self, lines, latent_space=None, batch_size=None, cpu=True, seed=None
    ):
        """Invert spectral lines. The batch size is inferred from the length of the
        line arrays (i.e. one latent sample per line), or can be used to set the
        number latent draws for a _single_ observation.

        lines : dict of str to array-like
            The transformed lines (see `transform_lines`).
        latent_space : array-like, optional
            Fixed values to use for the latent space. Default: Draw new random samples.
        batch_size : int, optional
            The number of samples to draw if only one line profiles was
            provided. Default: infer from shape of lines.
        cpu : bool, optional
            Whether to return the final data to the cpu. Default: True
        seed : int, optional
            Manual random seed for PyTorch to control the latent space draws.
            Default: None (do not adjust the random state).

        Returns
        -------
        Tensor containing the associated packed atmospheres (see
        `inv_transform_atmosphere`).
        """
        if batch_size is None:
            batch_size = lines[self.line_profiles[0]].shape[0]
        elif (lines[self.line_profiles[0]].ndim == 2
              and batch_size != lines[self.line_profiles[0]].shape[0]
              and lines[self.line_profiles[0]].shape[0] != 1):
            raise ValueError("`batch_size` should match the number of observations, if more than one is passed")

        if seed is not None:
            torch.manual_seed(seed)

        if latent_space is None:
            latent_space = torch.randn(batch_size, self.model.latent_size)

        input = torch.zeros(batch_size, self.model.size)
        input[:, : self.model.latent_size] = latent_space
        for idx, line in enumerate(self.line_profiles):
            line_slice = self.line_slice(idx)
            input[:, line_slice] = lines[line]

        input = input.to(self.dev)
        with torch.no_grad():
            result = self.model(input, rev=True)[0]
        if cpu:
            result = result.cpu()

        return result

    def invert_dual_cubes(
        self,
        crisp_seq: CRISPSequence,
        batch_size=256,
        latent_draws=128,
        seed=None,
        rotate=False,
        crispy_line_mapping=None,
        progress=True,
        inverse_trans_ignore=None
    ):
        """Invert observations of spectral lines as loaded by the sst-crispy package.

        Parameters
        ----------
        crisp_seq : CRISPSequence
            Sequence of spectral lines in the order expected by the model (see `self.line_profiles`).
        batch_size : int, optional
            The number of samples to put through the network at once. Default:
            256 (quite conservative). Must be a multiple of `latent_draws`.
        latent_draws : int, optional
            The number of latent draws for the inversion of each pixel. Default: 128.
        seed : int, optional
            Manual random seed for PyTorch to control the latent space draws.
            Default: None (do not adjust the random state).
        rotate : bool, optional
            Whether to use the `rotate_crop` method of CRISPSequence prior to
            inverting. Default: False.
        crispy_line_mapping : dict of str to str, optional
            Map of strs used for line profile names by the Radynversion model to
            those used by Crispy. Default works with default pretrained
            Radynversion models.
        progress : bool, optional
            Whether to show a progress bar for the inversion. Default: True.
        inverse_trans_ignore : list of str, optional
            Atmospheric parameters to ignore the when inverse transforming the
            inverted atmospheres (passed to `inv_transform_atmosphere`). Default
            works with default pretrained Radynversion models.

        Returns
        -------
        `crispy.Inversion` containing the inverted data including errors in the
        form of median absolute deviation.
        """
        if not isinstance(crisp_seq, CRISPSequence):
            raise ValueError(
                "`crisp_seq` expected to be a `crsipy.CRISPSequence` of the data in order of `RadynversionAdapter.line_profiles`."
            )

        if len(self.line_profiles) != 2:
            raise ValueError(
                "This function is designed for a two-line Radynversion, due to the image-coalignment method in crispy"
            )

        if batch_size / latent_draws != batch_size / latent_draws:
            raise ValueError(
                "`batch_size` should be an integer multiple of `latent_draws`"
            )

        if rotate:
            crisp_seq.rotate_crop()
        line_a, line_b = crisp_seq.data
        if line_a.dtype is not np.float32:
            line_a = line_a.astype(np.float32)
            line_b = line_b.astype(np.float32)

        if inverse_trans_ignore is None:
            inverse_trans_ignore = ['ne', 'temperature']

        delta_lambdas = {
            line: (crisp_seq.wvls[i] - np.median(crisp_seq.wvls[i])).value
            for i, line in enumerate(self.line_profiles)
        }
        im_shape = (line_a.shape[-2], line_a.shape[-1])
        line_a = np.ascontiguousarray(line_a.reshape(line_a.shape[0], -1).T)
        line_b = np.ascontiguousarray(line_b.reshape(line_b.shape[0], -1).T)

        lines = {self.line_profiles[0]: line_a, self.line_profiles[1]: line_b}
        transformed_lines = self.transform_lines(lines, delta_lambdas)

        if seed is not None:
            torch.manual_seed(seed)

        results = {
            param: np.zeros((self.model.atmos_size, line_a.shape[0]), dtype=np.float32)
            for param in self.atmos_params
        }
        for line in self.line_profiles:
            results[line] = np.zeros(
                (self.model.line_profile_size, line_a.shape[0]), dtype=np.float32
            )
        results["mad"] = np.zeros(
            (self.model.atmos_size, line_a.shape[0], len(self.atmos_params)),
            dtype=np.float32,
        )
        results["spect_mad"] = np.zeros(
            (self.model.line_profile_size, line_a.shape[0], len(self.line_profiles)),
            dtype=np.float32,
        )

        num_batches = int(np.ceil((line_a.shape[0] * latent_draws) / batch_size))
        if progress:
            batch_iter = tqdm(range(num_batches))
        else:
            batch_iter = range(num_batches)
        for i in batch_iter:
            start = i * batch_size
            end = (i + 1) * batch_size

            pixel_start = start // latent_draws
            pixel_end = end // latent_draws
            if pixel_end == pixel_start:
                pixel_end = pixel_start + 1

            curr_batch_size = batch_size
            if i == num_batches - 1:
                end = None
                curr_batch_size = (line_a.shape[0] * latent_draws) - start

            with torch.no_grad():
                yz = torch.zeros((curr_batch_size, self.model.size))
                yz[:, : self.model.latent_size] = torch.randn(
                    curr_batch_size, self.model.latent_size
                )

                for l_idx, line in enumerate(self.line_profiles):
                    line_slice = self.line_slice(l_idx)
                    for pix_idx, px in enumerate(range(pixel_start, pixel_end)):
                        yz[pix_idx * latent_draws:(pix_idx + 1) * latent_draws, line_slice] = transformed_lines[line][px]

                x_out = self.model(yz.to(self.dev), rev=True)[0]
                x_out[:, self.model.atmos_size * self.model.num_atmos_params :] = 0.0
                y_round_trip = self.model(x_out)[0]
                y_round_trip = y_round_trip.cpu().numpy()

                atmos_out = self.inv_transform_atmosphere(x_out, ignore=inverse_trans_ignore)
                atmos_out = {
                    k: v.reshape(-1, latent_draws, self.model.atmos_size).cpu().numpy()
                    for k, v in atmos_out.items()
                }

            for p_idx, param in enumerate(self.atmos_params):
                results[param][:, pixel_start:pixel_end] = np.swapaxes(
                    np.median(atmos_out[param], axis=1), 0, 1
                )
                results["mad"][:, pixel_start:pixel_end, p_idx] = np.swapaxes(
                    mad(atmos_out[param], axis=1), 0, 1
                )

            lines_out = {
                line: y_round_trip[:, self.line_slice(l_idx)].reshape(
                    -1, latent_draws, self.model.line_profile_size
                )
                for l_idx, line in enumerate(self.line_profiles)
            }

            for l_idx, line in enumerate(self.line_profiles):
                line_val = lines_out[line]
                results[line][:, pixel_start:pixel_end] = np.swapaxes(
                    np.median(line_val, axis=1), 0, 1
                )
                results["spect_mad"][:, pixel_start:pixel_end, l_idx] = np.swapaxes(
                    mad(line_val, axis=1), 0, 1
                )

        obj_dict = ObjDict()
        for p_idx, param in enumerate(self.atmos_params):
            obj_dict[param] = results[param].reshape(self.model.atmos_size, *im_shape)
            obj_dict[f"{param}_err"] = results["mad"][:, :, p_idx].reshape(self.model.atmos_size, *im_shape)

        if crispy_line_mapping is None:
            crispy_line_mapping = {"Halpha": "Halpha", "CaII8542": "Ca8542"}

        for l_idx, line in enumerate(self.line_profiles):
            crispy_line = crispy_line_mapping[line]
            obj_dict[f"{crispy_line}_true"] = transformed_lines[line].T.reshape(self.model.line_profile_size, *im_shape).numpy()
            obj_dict[f"{crispy_line}"] = results[line].reshape(self.model.line_profile_size, *im_shape)
            obj_dict[f"{crispy_line}_err"] = results["spect_mad"][:, :, l_idx].reshape(
                self.model.line_profile_size,
                *im_shape
            )

        inversion = Inversion(
            obj_dict,
            crisp_seq.header[0],
            self.z_stratification,
        )
        return inversion
