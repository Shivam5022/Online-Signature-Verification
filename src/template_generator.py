import numpy as np

class TemplateGenerator:
    """Generates user templates from enrolled signatures"""
    
    def __init__(self, beta=1.5):
        """
        Args:
            beta: Scaling factor for quantization step size
        """
        self.beta = beta

    def generate_template(self, enrolled_features):
        """
        Creates user template and quantization parameters
        Args:
            enrolled_features: List of feature vectors from enrollment signatures
        Returns:
            tuple: (template vector, quantization steps)
        """
        enrolled = np.array(enrolled_features)
        
        # Compute mean and standard deviation of each feature
        mu = np.mean(enrolled, axis=0)
        sigma = np.std(enrolled, axis=0)
        
        # Compute quantization step sizes
        q = self.beta * sigma
        
        # Add small epsilon to prevent division by zero
        # Different epsilons for absolute vs relative frequency histograms
        epsilon = np.where(np.arange(len(q)) < 100, 0.002, 0.8)
        q += epsilon
        
        # Quantize enrolled features and compute template as mean
        quantized = enrolled / q
        template = np.mean(quantized, axis=0)
        
        return template, q
