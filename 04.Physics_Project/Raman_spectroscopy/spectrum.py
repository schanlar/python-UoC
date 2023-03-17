import numpy as np
import matplotlib.pyplot as plt
try:
    from typing import List, Tuple, Optional
except ImportError:
    os.system("pip install typing")
    from typing import List, Tuple, Optional

class Spectrum():
    def __init__(self,
                 wavelengths: np.ndarray,
                 n_components: Optional[int] = None,
                 n_peaks: Optional[List[int]] = None,
                 concentration: Optional[List[float]] = None
    ) -> None:
        
        self._wavelengths = wavelengths
        self._n_components = n_components
        self._n_peaks = n_peaks
        self._concentration = concentration
        return None

    
    def make(self,
             add_noise: bool = False,
             add_baseline: bool = False,
             add_spikes: bool = False,
             seed: int = 10
    ) -> np.ndarray:
        
        mix_spectrum = np.zeros_like(self._wavelengths)
        
        if self._n_components is None:
            return mix_spectrum
        
        else:
            np.random.seed(seed) # for reproducibility
            
            A, B, C = self._add_components(n_components = self._n_components, n_peaks = self._n_peaks)
            if (self._concentration is None):
                self._concentration  = [0.5, 0.3, 0.2]
                print("WARNING: initial configuration of spectrum did not include concentrations")
                print("--> random concentrations added:", self._concentration)
                
            for c, s in zip(self._concentration, [A, B, C]):
                mix_spectrum = mix_spectrum + (c * s)
                
            if add_noise:
                # Random noise:
                mix_spectrum = mix_spectrum +  np.random.normal(0, 0.02, len(self._wavelengths))
                print("Random noise added to spectrum: Done")
                
            if add_baseline:
                # Baseline as a polynomial background:
                poly = (0.2 * np.ones(len(self._wavelengths))) + \
                       (0.0001 * self._wavelengths) + \
                       (0.000051 * (self._wavelengths - 680)**2)
                mix_spectrum = mix_spectrum + poly
                print("Baseline added to spectrum: Done")
                
            if add_spikes:
                for n_spikes in range(1, np.random.randint(2, 8)):
                    pos = np.random.randint(0, len(mix_spectrum)+1)
                    mix_spectrum[pos] = mix_spectrum[pos] + (np.random.randint(3, 20) / 10.)

                print("Cosmic ray spikes added to spectrum: Done ({} spikes)".format(n_spikes))
                
            return mix_spectrum
    
    
    def _gauss(self,
               x: np.ndarray,
               mu: float,
               sigma: float,
               A: float = 1.
    ) -> np.ndarray:
        """
        A Gaussian fit function
        """
        return A * np.exp(-(x-mu)**2 / sigma**2)

    
    def _add_components(self,n_components: int = 3,
                        n_peaks: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, ...]:
        
        assert n_components in [1,2,3], "Number of components can be 1, 2, or 3"
        gauss_a = np.zeros_like(self._wavelengths)
        spectrum_a = np.zeros_like(gauss_a)
        
        gauss_b = np.zeros_like(self._wavelengths)
        spectrum_b = np.zeros_like(gauss_b)
        
        gauss_c = np.zeros_like(self._wavelengths)
        spectrum_c = np.zeros_like(gauss_c)
        
        if (n_components == 1) and (n_peaks is None):
            
            # Component A with 3 gaussians
            gauss_a =  self._gauss(x=self._wavelengths, mu=663, sigma=1., A=1.) + \
                       self._gauss(x=self._wavelengths, mu=735, sigma=1., A=.2) + \
                       self._gauss(x=self._wavelengths, mu=771, sigma=1., A=.3)
            
            # Component normalization
            spectrum_a = gauss_a / np.max(gauss_a)
            
        elif (n_components == 2) and (n_peaks is None):
            # Component A with 3 gaussians
            gauss_a =  self._gauss(x=self._wavelengths, mu=663, sigma=1., A=1.) + \
                       self._gauss(x=self._wavelengths, mu=735, sigma=1., A=.2) + \
                       self._gauss(x=self._wavelengths, mu=771, sigma=1., A=.3)

            # Component B with 4 gaussians
            gauss_b = self._gauss(x=self._wavelengths, mu=700, sigma=1., A=.2) + \
                      self._gauss(x=self._wavelengths, mu=690, sigma=2., A=.5) + \
                      self._gauss(x=self._wavelengths, mu=710, sigma=1., A=.75) + \
                      self._gauss(x=self._wavelengths, mu=774, sigma=1.5, A=.25)
            
            # Component normalization
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            
        elif (n_components == 3) and (n_peaks is None):
            # Component A with 3 gaussians
            gauss_a =  self._gauss(x=self._wavelengths, mu=663, sigma=1., A=1.) + \
                       self._gauss(x=self._wavelengths, mu=735, sigma=1., A=.2) + \
                       self._gauss(x=self._wavelengths, mu=771, sigma=1., A=.3)

            # Component B with 4 gaussians
            gauss_b = self._gauss(x=self._wavelengths, mu=700, sigma=1., A=.2) + \
                      self._gauss(x=self._wavelengths, mu=690, sigma=2., A=.5) + \
                      self._gauss(x=self._wavelengths, mu=710, sigma=1., A=.75) + \
                      self._gauss(x=self._wavelengths, mu=774, sigma=1.5, A=.25)

            # Component C with 2 gaussians
            gauss_c = self._gauss(x=self._wavelengths, mu=660, sigma=1., A=.05) + \
                      self._gauss(x=self._wavelengths, mu=712, sigma=4., A=.7)

            # Component normalization
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            spectrum_c = gauss_c / np.max(gauss_c)
            
        elif (n_components == 1) and (n_peaks is not None):
            for _ in range(n_peaks[0]):
                gauss_a = gauss_a + self._gauss(x=self._wavelengths,
                                 mu=np.random.randint(int(np.min(self._wavelengths)), int(np.max(self._wavelengths))),
                                 sigma=np.random.randint(1, 5),
                                 A=np.random.randint(1, 11) / 10.)
            spectrum_a = gauss_a / np.max(gauss_a)
            
        elif (n_components == 2) and (n_peaks is not None):
            for _ in range(n_peaks[0]):
                gauss_a = gauss_a + self._gauss(x=self._wavelengths,
                                 mu=np.random.randint(int(np.min(self._wavelengths)), int(np.max(self._wavelengths))),
                                 sigma=np.random.randint(1, 5),
                                 A=np.random.randint(1, 11) / 10.)
                
            for _ in range(n_peaks[1]):    
                gauss_b = gauss_b + self._gauss(x=self._wavelengths,
                                 mu=np.random.randint(int(np.min(self._wavelengths)), int(np.max(self._wavelengths))),
                                 sigma=np.random.randint(1, 5),
                                 A=np.random.randint(1, 11) / 10.)
                
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            
        elif (n_components == 3) and (n_peaks is not None):
            for _ in range(n_peaks[0]):
                gauss_a = gauss_a + self._gauss(x=self._wavelengths,
                                 mu=np.random.randint(int(np.min(self._wavelengths)), int(np.max(self._wavelengths))),
                                 sigma=np.random.randint(1, 5),
                                 A=np.random.randint(1, 11) / 10.)
                
            for _ in range(n_peaks[1]):         
                gauss_b = gauss_b + self._gauss(x=self._wavelengths,
                                 mu=np.random.randint(int(np.min(self._wavelengths)), int(np.max(self._wavelengths))),
                                 sigma=np.random.randint(1, 5),
                                 A=np.random.randint(1, 11) / 10.)
                
            for _ in range(n_peaks[2]):
                gauss_c = gauss_c + self._gauss(x=self._wavelengths,
                                 mu=np.random.randint(int(np.min(self._wavelengths)), int(np.max(self._wavelengths))),
                                 sigma=np.random.randint(1, 5),
                                 A=np.random.randint(1, 11) / 10.)
                
            spectrum_a = gauss_a / np.max(gauss_a)
            spectrum_b = gauss_b / np.max(gauss_b)
            spectrum_c = gauss_c / np.max(gauss_c)
        
        return spectrum_a, spectrum_b, spectrum_c
    
    def plot_components(self) -> None:
        if self._n_components is None:
            raise ValueError("There are no components to plot")
            
        labels = ["Component A", "Component B", "Component C"]
       
        for idx, val in enumerate(range(self._n_components)):
            plt.plot(self._wavelengths,
                     self._add_components(n_components = self._n_components, n_peaks = self._n_peaks)[idx],
                     label=labels[idx])
        
        plt.title('Known components in our mixture', fontsize=15)
        plt.xlabel('Wavelength', fontsize=15)
        plt.ylabel('Normalized intensity', fontsize=15)

        plt.legend()
        plt.show()
        return None
    
    
if __name__ == "__main__":
    pass