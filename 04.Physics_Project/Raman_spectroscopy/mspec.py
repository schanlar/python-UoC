import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
try:
    from typing import List, Tuple, Optional, Union
except ImportError:
    os.system("pip install typing")
    from typing import List, Tuple, Optional, Union
    
@dataclass
class SpectrumConfig:
    """
        A dataclass to provide the basic initial configuration for a spectrum.

        Data fields
        ===========

            wavelengths     :   array-like
                                    the wavelength range for the spectrum.
            n_components    :   int or None, default None
                                    the number of individual components (max 3) that wil
                                    compose the final mixture spectrum.
            n_peaks         :   List[int] or None, default None
                                    the number of Gaussian peaks of each individual component
                                    spectrum.
            concentration   :   List[float] or None, default None
                                    the concentrations of each component in the final mixture.

    """
    wavelengths: np.ndarray
    n_components: Optional[int] = None
    n_peaks: Optional[List[int]] = None
    concentration: Optional[List[float]] = None

class Spectrum:
    """
        A class to generate a spectrum based on a mixture of up to three Gaussian components.
    """
    def __init__(self, config: SpectrumConfig) -> None:
        """
            Initialize self.  See help(type(self)) for accurate signature.

            Arguments
            =========
                config : __main__.SpectrumConfig
                            initial configuration of the spectrum. It includes characterstics
                            such as the wavelength range, the number of components (max 3) that
                            will compose the final mixture spectrum, the number of Gaussian peaks
                            of each individual component spectrum, and the concentrations of each
                            component in the final mixture.
        """
        self._config = config
        return None

    
    def to_array(self,
                add_noise: bool = False,
                add_baseline: bool = False,
                add_spikes: bool = False,
                seed: int = 10,
                verbose: bool = False
    ) -> np.ndarray:
        """
            Arguments
            =========
                add_noise    : bool, default False
                                if True add random noise to simulate contributions 
                                from the measurement device.
                add_baseline : bool, default False 
                                if True add a polynomial baseline to simulate contributions 
                                from fluorescence background.
                add_spikes   : bool, default False
                                if True add spikes at random wavelengths to simulate contributions
                                from cosmic rays. The number of spikes can be controlled, indirectly, 
                                by the "seed" argument (see below).
                seed         : int, default 10
                                seed for random number generator.
                verbose      : bool, default False
                                if True print a confirmation message when a source of noise is added to spectrum.

            Returns
            =======
                mix_spectrum : array-like
                                it returns a 1D numpy array of the intensity of the spectrum
        """
        
        mix_spectrum = np.zeros_like(self._config.wavelengths)
        
        if self._config.n_components is None:
            return mix_spectrum
        
        else:
            np.random.seed(seed) # for reproducibility
            
            A, B, C = self._add_components(n_components = self._config.n_components, n_peaks = self._config.n_peaks)
            if (self._config.concentration is None):
                self._config.concentration  = [0.5, 0.3, 0.2]
                print("WARNING: initial configuration of spectrum did not include concentrations")
                print("--> random concentrations added:", self._config.concentration)
                
            for c, s in zip(self._config.concentration, [A, B, C]):
                mix_spectrum = mix_spectrum + (c * s)
                
            if add_noise:
                # Random noise:
                mix_spectrum = mix_spectrum +  np.random.normal(0, 0.02, len(self._config.wavelengths))
                if verbose: print("Random noise added to spectrum: Done")
                
            if add_baseline:
                # Baseline as a polynomial background:
                poly = (0.2 * np.ones(len(self._config.wavelengths))) + \
                       (0.0001 * self._config.wavelengths) + \
                       (0.000051 * (self._config.wavelengths - 680)**2)
                mix_spectrum = mix_spectrum + poly
                if verbose: print("Baseline added to spectrum: Done")
                
            if add_spikes:
                for n_spikes in range(1, np.random.randint(2, 8)):
                    pos = np.random.randint(0, len(mix_spectrum)+1)
                    mix_spectrum[pos] = mix_spectrum[pos] + (np.random.randint(3, 20) / 10.)

                if verbose: print("Cosmic ray spikes added to spectrum: Done ({} spikes)".format(n_spikes))
                
            return mix_spectrum
    
    def _gauss(self,
               x: Union[int, float, np.ndarray],
               mu: Union[int, float],
               sigma: Union[int, float],
               A: Union[int, float] = 1.
    ) -> np.ndarray:
        """
            A Gaussian function

            Arguments
            =========
                x     : int or float or array-like
                            the range for the x-axis
                mu    : int or float
                            the position of the center of the peak
                sigma : int or float
                            the standard deviation
                A     : int or float, default 1.0
                            the height of the curve's peak

            Returns
            =======
                array-like : the Gaussian distribution 
        """
        return A * np.exp(-(x-mu)**2 / sigma**2)

    def _add_components(self,
                        n_components: int = 3,
                        n_peaks: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, ...]:
        """
            Arguments
            =========
                n_components : int, default 3
                                    the number of components in the final mixture (max 3)
                n_peaks      : List[int] or None, default None
                                    the number of Gaussian peaks in each individual component

            Returns
            =======
                spectrum_a, spectrum_b, spectrum_c : Tuple[array-like, ...]
                                                        the normalized intensity of each component 
        """
        
        assert n_components in [1,2,3], "Number of components can be 1, 2, or 3"
        gauss_a = np.zeros_like(self._config.wavelengths)
        spectrum_a = np.zeros_like(gauss_a)
        
        gauss_b = np.zeros_like(self._config.wavelengths)
        spectrum_b = np.zeros_like(gauss_b)
        
        gauss_c = np.zeros_like(self._config.wavelengths)
        spectrum_c = np.zeros_like(gauss_c)
        
        if (n_components == 1) and (n_peaks is None):
            
            # default: component with 3 gaussians
            gauss_a =  self._gauss(x=self._config.wavelengths, mu=663, sigma=1., A=1.) + \
                       self._gauss(x=self._config.wavelengths, mu=735, sigma=1., A=.2) + \
                       self._gauss(x=self._config.wavelengths, mu=771, sigma=1., A=.3)
            
            # component normalization
            spectrum_a = gauss_a / np.max(gauss_a)
            
        elif (n_components == 2) and (n_peaks is None):
            # default: component A with 3 gaussians
            gauss_a =  self._gauss(x=self._config.wavelengths, mu=663, sigma=1., A=1.) + \
                       self._gauss(x=self._config.wavelengths, mu=735, sigma=1., A=.2) + \
                       self._gauss(x=self._config.wavelengths, mu=771, sigma=1., A=.3)

            # default: component B with 4 gaussians
            gauss_b = self._gauss(x=self._config.wavelengths, mu=700, sigma=1., A=.2) + \
                      self._gauss(x=self._config.wavelengths, mu=690, sigma=2., A=.5) + \
                      self._gauss(x=self._config.wavelengths, mu=710, sigma=1., A=.75) + \
                      self._gauss(x=self._config.wavelengths, mu=774, sigma=1.5, A=.25)
            
            # component normalization
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            
        elif (n_components == 3) and (n_peaks is None):
            # default: component A with 3 gaussians
            gauss_a =  self._gauss(x=self._config.wavelengths, mu=663, sigma=1., A=1.) + \
                       self._gauss(x=self._config.wavelengths, mu=735, sigma=1., A=.2) + \
                       self._gauss(x=self._config.wavelengths, mu=771, sigma=1., A=.3)

            # default: component B with 4 gaussians
            gauss_b = self._gauss(x=self._config.wavelengths, mu=700, sigma=1., A=.2) + \
                      self._gauss(x=self._config.wavelengths, mu=690, sigma=2., A=.5) + \
                      self._gauss(x=self._config.wavelengths, mu=710, sigma=1., A=.75) + \
                      self._gauss(x=self._config.wavelengths, mu=774, sigma=1.5, A=.25)

            # default: component C with 2 gaussians
            gauss_c = self._gauss(x=self._config.wavelengths, mu=660, sigma=1., A=.05) + \
                      self._gauss(x=self._config.wavelengths, mu=712, sigma=4., A=.7)

            # component normalization
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            spectrum_c = gauss_c / np.max(gauss_c)
            
        elif (n_components == 1) and (n_peaks is not None):
            for _ in range(n_peaks[0]):
                gauss_a = gauss_a + self._gauss(
                    x=self._config.wavelengths,
                    mu=np.random.randint(int(np.min(self._config.wavelengths)), int(np.max(self._config.wavelengths))),
                    sigma=np.random.randint(1, 5),
                    A=np.random.randint(1, 11) / 10.
                    )
            spectrum_a = gauss_a / np.max(gauss_a)
            
        elif (n_components == 2) and (n_peaks is not None):
            for _ in range(n_peaks[0]):
                gauss_a = gauss_a + self._gauss(
                    x=self._config.wavelengths,
                    mu=np.random.randint(int(np.min(self._config.wavelengths)), int(np.max(self._config.wavelengths))),
                    sigma=np.random.randint(1, 5),
                    A=np.random.randint(1, 11) / 10.
                    )
                
            for _ in range(n_peaks[1]):    
                gauss_b = gauss_b + self._gauss(
                    x=self._config.wavelengths,
                    mu=np.random.randint(int(np.min(self._config.wavelengths)), int(np.max(self._config.wavelengths))),
                    sigma=np.random.randint(1, 5),
                    A=np.random.randint(1, 11) / 10.
                    )
                
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            
        elif (n_components == 3) and (n_peaks is not None):
            for _ in range(n_peaks[0]):
                gauss_a = gauss_a + self._gauss(
                    x=self._config.wavelengths,
                    mu=np.random.randint(int(np.min(self._config.wavelengths)), int(np.max(self._config.wavelengths))),
                    sigma=np.random.randint(1, 5),
                    A=np.random.randint(1, 11) / 10.
                    )
                
            for _ in range(n_peaks[1]):         
                gauss_b = gauss_b + self._gauss(
                    x=self._config.wavelengths,
                    mu=np.random.randint(int(np.min(self._config.wavelengths)), int(np.max(self._config.wavelengths))),
                    sigma=np.random.randint(1, 5),
                    A=np.random.randint(1, 11) / 10.
                    )
                
            for _ in range(n_peaks[2]):
                gauss_c = gauss_c + self._gauss(
                    x=self._config.wavelengths,
                    mu=np.random.randint(int(np.min(self._config.wavelengths)), int(np.max(self._config.wavelengths))),
                    sigma=np.random.randint(1, 5),
                    A=np.random.randint(1, 11) / 10.
                    )
                
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            spectrum_c = gauss_c / np.max(gauss_c)
        
        return spectrum_a, spectrum_b, spectrum_c
    
    def plot_components(self) -> None:
        """
            A function that displays the spectrum of
            each individual component of the mixture

            Arguments
            =========
                None

            Returns
            =======
                None
        """
        if self._config.n_components is None:
            raise ValueError("There are no components to plot")
            
        labels = ["Component A", "Component B", "Component C"]
       
        for idx, val in enumerate(range(self._config.n_components)):
            plt.plot(self._config.wavelengths,
                     self._add_components(n_components = self._config.n_components, n_peaks = self._config.n_peaks)[idx],
                     label=labels[idx])
        
        plt.title('Known components in our mixture', fontsize=12)
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Normalized intensity (arb. unit)', fontsize=12)

        plt.legend()
        plt.show()
        return None
    
    def plot_spectrum(self,
                      color: str = "black",
                      add_noise: bool = False,
                      add_baseline: bool = False,
                      add_spikes: bool = False,
                      seed: int = 10
    ) -> None:
        """
            A function that displays the mixture spectrum

            Arguments
            =========
                color        : str, default "black"
                                the color of spectrum
                add_noise    : bool, default False
                                if True add and display random noise stem to simulate contributions 
                                from the measurement device.
                add_baseline : bool, default False
                                if True add and display a polynomial baseline to simulate contributions 
                                from fluorescence background.
                add_spikes   : bool, default False
                                if True add and display spikes at random wavelengths to simulate contributions
                                from cosmic rays.
                seed         : int, default 10
                                seed for random number generator.

            Returns
            =======
                None
        """
        if self._config.n_components is None:
            raise ValueError("There are no components to plot")
        if (add_noise or add_baseline or add_spikes):
            title = "Mixture spectrum with noise"
        else:
            title = "Mixture spectrum"
            
        y = self.to_array(add_noise=add_noise, add_baseline=add_baseline, add_spikes=add_spikes, seed=seed)
        plt.plot(self._config.wavelengths, y, color=color)
        
        plt.title(title, fontsize=12)
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Intensity (arb. unit)', fontsize=12)

        plt.show()
        return None
    

if __name__ == "__main__":
    help(SpectrumConfig)
    help(Spectrum)
    