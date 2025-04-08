import os

class SignatureReader:
    """Reads the signature data from the files"""
    
    @staticmethod
    def read_signature(file_path):
        """
        Returns:
            tuple: (X-coords, Y-coords, Pressure values)
        """
        with open(file_path, 'r') as f:
            num_points = int(f.readline().strip())
            X, Y, P = [], [], []
            for _ in range(num_points):
                parts = f.readline().strip().split()
                X.append(float(parts[0]))  # X coordinate
                Y.append(float(parts[1]))  # Y coordinate
                P.append(float(parts[6]))  # Pressure value
                
        return X, Y, P

    @staticmethod
    def get_user_files(data_dir, user_id, genuine=True):
        """
        Gets all signature files for a specific user
        Args:
            data_dir: Directory containing signature files
            user_id: User number (1-5)
            genuine: Whether to get genuine (True) or forged (False) signatures
        Returns:
            list: Sorted list of file paths
        """
        prefix = f"USER{user_id}_"
        
        files = [f for f in os.listdir(data_dir) if f.startswith(prefix)]
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
         
        # Return first 20 for genuine, last 20 for forged
        return files[:20] if genuine else files[20:40]
