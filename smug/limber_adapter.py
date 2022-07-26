import astropy.units as u
import numpy as np
import torch
from weno4 import weno4


class LimberAdapter:
    """Prepare some data and run a Limber model on it.

    Parameters
    ----------
    model : LimberModel
        The model to use (see `limber_model.pretrained_limber`
    line_grid : array-like
        The wavelength grid used in the model.
    dev : torch.device on which to run the model, optional
        The device to use (default: gpu if available, otherwise cpu).
    """

    def __init__(self, model, line_grid, dev=None):

        self.model = model
        self.line_grid = (line_grid << u.Angstrom).value
        if dev is None:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dev = dev
        self.model.to(dev)
        self.model.eval()

    @staticmethod
    def to_tensor(x):
        return torch.from_numpy(x).float()

    def reproject_data(
        self,
        image,
        wavelength,
        mu_observed,
        interp=None,
        batch_size=1024,
        reconstruct_original_shape=False,
        cpu=True,
    ):
        """
        Use the limber model to reproject data from the provided viewing angle back to mu=1.

        Parameters
        ----------
        image : array-like
            The image to rotate, expected axes [wavelength, y, x].
        wavelength : array-like or astropy.Quantity
            The wavelength axis for the image. Will be converted to Angstrom if
            astropy.Quantity, otherwise assumed to be in Angstrom.
        mu_observed : float
            The cosine of the viewing angle of the observation.
        interp : Callable[[array-like, array-like, array-like], array-like], optional
            The interpolation function to use for up/downsampling the line
            profiles. Takes the same signature as `np.interp` (default: weno4).
        batch_size : int, optional
            The batch size to use (default: 1024).
        reconstruct_original_shape : bool, optional
            Whether to return data in the original shape [original_wavelengths,
            y, x]. or the internal shape used for the network [y, x,
            interpolated_wavelengths] (default: False).
        cpu : bool, optional
            Whether to return the data to the CPU (default: True), can be useful
            to keep the data on GPU (as a torch.Tensor) if it's to be used at
            later stages of a pipeline. `reconstruct_original_shape` can only be
            set if `cpu` is True. If true, result is returned as a np.array.

        Returns
        -------
        data : array
            Three-dimensional array of the reconstructed data, the axis order of
            which depends on the result of `reconstruct_original_shape` and
            `cpu`. If `cpu` is False, a `torch.Tensor` will be returned.
        """
        if interp is None:
            interp = weno4

        if reconstruct_original_shape and not cpu:
            raise ValueError(
                "Cannot interpolate back to original wavelengths "
                "(`reconstruct_original_shape`) without returning tensors to cpu"
            )

        flat_im = np.ascontiguousarray(image.reshape(image.shape[0], -1).T).astype(
            "<f4"
        )
        flat_im_upsample = np.zeros(
            (flat_im.shape[0], self.model.size), dtype=np.float32
        )

        wavelength = (wavelength << u.Angstrom).value
        for i in range(flat_im_upsample.shape[0]):
            flat_im_upsample[i, 1:] = interp(self.line_grid, wavelength, flat_im[i])
        flat_im_upsample[:, 0] = mu_observed

        cont_value = np.copy(flat_im_upsample[:, 1])
        flat_im_upsample[:, 1:] /= cont_value[:, None]

        num_batches = int(np.ceil(flat_im_upsample.shape[0] / batch_size))

        im_in = self.to_tensor(flat_im_upsample).to(self.dev)
        im_out = torch.empty_like(im_in)
        with torch.no_grad():
            for i in range(num_batches):
                sl = slice(i * batch_size, min((i + 1) * batch_size, im_in.shape[0]))
                im_out = self.model(im_in[sl])
        cont_gpu = self.to_tensor(cont_value).to(self.dev)
        im_out[:, 1:] *= cont_gpu[:, None]

        if cpu:
            im_out = im_out.cpu()

        if reconstruct_original_shape:
            im_out = im_out.numpy()
            im_out_downsampled = np.empty((im_out.shape[0], image.shape[0]))
            for i in range(im_out.shape[0]):
                im_out_downsampled[i, :] = weno4(
                    wavelength, self.line_grid, im_out[i, 1:]
                )
            return im_out_downsampled.T.reshape(image.shape)

        return im_out.reshape(*image.shape[1:], -1).numpy()
