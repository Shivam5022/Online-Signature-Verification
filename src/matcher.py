import numpy as np

class Matcher:
    """Handles signature verification using Manhattan distance"""
    
    @staticmethod
    def verify_signature(test_feature, template, q):
        """
        Computes dissimilarity score between test signature and template
        Args:
            test_feature: Feature vector of test signature
            template: User's template vector
            q: Quantization step sizes
        Returns:
            float: Dissimilarity score (lower means more similar)
        """
        # Quantize test features using same steps as template
        quantized_test = test_feature / q
        
        # Compute Manhattan distance between test and template
        return np.sum(np.abs(quantized_test - template))
