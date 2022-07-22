import astropy.units as u
import numpy as np
import torch
from weno4 import weno4


class LimberAdapter:
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
        reconstruct_original_wavelengths=False,
        cpu=True,
    ):
        if interp is None:
            interp = weno4

        if reconstruct_original_wavelengths and not cpu:
            raise ValueError(
                "Cannot interpolate back to original wavelengths "
                "(`reconstruct_original_wavelengths`) without returning tensors to cpu"
            )

        flat_im = np.ascontiguousarray(image.reshape(image.shape[0], -1).T)
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

        if reconstruct_original_wavelengths:
            im_out = im_out.numpy()
            im_out_downsampled = np.empty_like(im_out)
            for i in range(im_out.shape[0]):
                im_out_downsampled[i, :] = weno4(
                    wavelength, self.line_grid, im_out[i, 1:]
                )
            return im_out_downsampled.T.reshape(image.shape)

        return im_out.reshape(*image.shape[1:], -1)
