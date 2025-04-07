import math
import numpy as np

class FeatureExtractor:
    """Extracts dynamic features from signature data"""
    
    def __init__(self):
        """Initialize with default histogram parameters"""
        # Number of bins for each histogram type
        self.angle_bins = 20
        self.speed_bins = 10
        self.coord_bins = 20
        self.pressure_bins = 10

    def compute_derivatives(self, data, max_order=2):
        """
        Computes derivatives using finite differences
        Args:
            data: Input time series data
            max_order: Highest derivative to compute
        Returns:
            dict: Derivatives of each order (1 to max_order)
        """
        derivatives = {}
        current = data
        
        for k in range(1, max_order+1):
            # Compute k-th order derivative using forward differences
            deriv = [current[i+1] - current[i] for i in range(len(current)-1)]
            derivatives[k] = deriv
            current = deriv
            
        return derivatives

    def compute_polar_coords(self, x_deriv, y_deriv):
        """
        Converts Cartesian coordinates to polar (r, theta)
        Args:
            x_deriv: X derivatives
            y_deriv: Y derivatives
        Returns:
            tuple: (radius, angle) arrays
        """
        radius = []
        angle = []
        
        for x, y in zip(x_deriv, y_deriv):
            radius.append(math.hypot(x, y))  # r = sqrt(x² + y²)
            angle.append(math.atan2(y, x))   # θ = arctan(y/x)
            
        return radius, angle

    def _hist_1d(self, data, bins, min_val, max_val, freq_type, sigma_factor=0):
        """
        Computes 1D histogram with optional dynamic range
        Args:
            data: Input data to bin
            bins: Number of bins
            min_val: Minimum bin value
            max_val: Maximum bin value
            freq_type: 'absolute' or 'relative' frequency
            sigma_factor: If >0, uses mean ± sigma_factor*std for range
        Returns:
            np.array: Histogram values
        """
        if not data:
            return np.zeros(bins)
            
        # Adjust range based on data statistics if requested
        if sigma_factor > 0:
            mu = np.mean(data)
            sigma = np.std(data)
            min_val = mu - sigma_factor * sigma
            max_val = mu + sigma_factor * sigma
            
        # Create bins and compute histogram
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        hist, _ = np.histogram(data, bins=bin_edges)
        
        # Normalize if relative frequency requested
        if freq_type == 'relative' and len(data) > 0:
            hist = hist / len(data)
            
        return hist

    def _hist_2d(self, x, y, x_bins, y_bins, x_min, x_max, y_min, y_max, 
                x_sigma=0, y_sigma=0, freq_type='absolute'):
        """
        Computes 2D histogram with optional dynamic ranges
        Args:
            x: First dimension data
            y: Second dimension data
            x_bins: Number of bins for x dimension
            y_bins: Number of bins for y dimension
            x_min: Minimum x value
            x_max: Maximum x value
            y_min: Minimum y value
            y_max: Maximum y value
            x_sigma: If >0, uses mean ± x_sigma*std for x range
            y_sigma: If >0, uses mean ± y_sigma*std for y range
            freq_type: 'absolute' or 'relative' frequency
        Returns:
            np.array: Flattened histogram values
        """
        if not x or not y:
            return np.zeros((x_bins) * (y_bins))
            
        # Adjust x range if requested
        if x_sigma > 0:
            x_mu, x_sig = np.mean(x), np.std(x)
            x_min, x_max = x_mu - x_sigma * x_sig, x_mu + x_sigma * x_sig
            
        # Adjust y range if requested
        if y_sigma > 0:
            y_mu, y_sig = np.mean(y), np.std(y)
            y_min, y_max = y_mu - y_sigma * y_sig, y_mu + y_sigma * y_sig
            
        # Create bins and compute 2D histogram
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)
        hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        
        # Normalize if relative frequency requested
        if freq_type == 'relative' and len(x) > 0:
            hist = hist / len(x)
            
        return hist.flatten()

    def process_signature(self, X, Y, P):
        """
        Main feature extraction pipeline
        Args:
            X: X coordinates
            Y: Y coordinates
            P: Pressure values
        Returns:
            np.array: Combined feature vector
        """
        # Compute derivatives up to order 2
        x_derivs = self.compute_derivatives(X)
        y_derivs = self.compute_derivatives(Y)
        p_derivs = self.compute_derivatives(P)
        
        # Get first and second derivatives
        x1, x2 = x_derivs.get(1, []), x_derivs.get(2, [])
        y1, y2 = y_derivs.get(1, []), y_derivs.get(2, [])
        p1, p2 = p_derivs.get(1, []), p_derivs.get(2, [])
        
        # Compute polar coordinates for first derivatives
        r1, theta1 = self.compute_polar_coords(x1, y1) if x1 and y1 else ([], [])
        
        # Initialize feature vector
        feature_vector = []
        
        # Histogram 1: Angle distribution (relative)
        hist = self._hist_1d(theta1, self.angle_bins, -np.pi, np.pi, 'relative')
        feature_vector.extend(hist)
        
        # Histogram 2: Angle derivative (relative)
        theta2_deriv = [theta1[i+1] - theta1[i] for i in range(len(theta1)-1)] if len(theta1) > 1 else []
        hist = self._hist_1d(theta2_deriv, self.angle_bins, -np.pi, np.pi, 'relative')
        feature_vector.extend(hist)
        
        # Histogram 3: Angle vs angle change (absolute)
        phi_d = [theta1[i+1] - theta1[i] for i in range(len(theta1)-1)]
        phi_d2 = [phi_d[i+1] - phi_d[i] for i in range(len(phi_d)-1)] if len(phi_d) > 1 else []
        theta1_valid = theta1[:-2] if len(theta1) >= 2 else []
        hist = self._hist_2d(theta1_valid, phi_d2, self.angle_bins, self.angle_bins, 
                            -np.pi, np.pi, -np.pi, np.pi, freq_type='absolute')
        feature_vector.extend(hist)
        
        # Histogram 4: Speed magnitude (absolute)
        hist = self._hist_1d(r1, self.speed_bins, 0, 0, 'absolute', sigma_factor=3)
        feature_vector.extend(hist)
        
        # Histogram 5: Speed derivative (absolute)
        r2, _ = self.compute_polar_coords(x2, y2) if x2 and y2 else ([], [])
        hist = self._hist_1d(r2, self.speed_bins, 0, 0, 'absolute', sigma_factor=3)
        feature_vector.extend(hist)
        
        # Histogram 6-9: Coordinate derivatives (relative)
        for data in [x1, y1, x2, y2]:
            hist = self._hist_1d(data, self.coord_bins, 0, 0, 'relative', sigma_factor=3)
            feature_vector.extend(hist)
        
        # Histogram 10-11: Coordinate derivative pairs (relative)
        for x, y in [(x1, x2), (y1, y2)]:
            # Ensure equal lengths
            min_len = min(len(x), len(y))
            x_valid = x[:min_len]
            y_valid = y[:min_len]
            hist = self._hist_2d(x_valid, y_valid, self.coord_bins//2, self.coord_bins//2,
                                0, 0, 0, 0, x_sigma=3, y_sigma=3, freq_type='relative')
            feature_vector.extend(hist)
        
        # Histogram 12-14: Angle-speed relationships (relative)
        for angle_data, speed_data in [(theta1, r1), (theta2_deriv, r2), (theta1, r2)]:
            # Ensure equal lengths and split into two halves
            min_len = min(len(angle_data), len(speed_data))
            angle_data = angle_data[:min_len]
            speed_data = speed_data[:min_len]
            split = min_len // 2
            
            # First half
            hist = self._hist_2d(angle_data[:split], speed_data[:split], 
                                self.angle_bins, self.speed_bins,
                                -np.pi, np.pi, 0, 0, y_sigma=3, freq_type='relative')
            feature_vector.extend(hist)
            
            # Second half
            hist = self._hist_2d(angle_data[split:], speed_data[split:],
                                self.angle_bins, self.speed_bins,
                                -np.pi, np.pi, 0, 0, y_sigma=3, freq_type='relative')
            feature_vector.extend(hist)
        
        # Histogram 15-16: Pressure features
        for p_data, bins, freq_type in [(p1, self.pressure_bins, 'absolute'),
                                      (p2, self.pressure_bins, 'relative')]:
            if not p_data:
                feature_vector.extend(np.zeros(bins*2))
                continue
                
            split = len(p_data) // 2
            # First half
            hist = self._hist_1d(p_data[:split], bins, 0, 0, freq_type, sigma_factor=3)
            feature_vector.extend(hist)
            # Second half
            hist = self._hist_1d(p_data[split:], bins, 0, 0, freq_type, sigma_factor=3)
            feature_vector.extend(hist)
        
        return np.array(feature_vector)
