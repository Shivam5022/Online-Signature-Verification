from signature_reader import SignatureReader
from feature_extractor import FeatureExtractor
from template_generator import TemplateGenerator
from matcher import Matcher
import os

def main():
    # Initialize all components
    print("Initializing system components...")
    reader = SignatureReader()
    extractor = FeatureExtractor()
    template_gen = TemplateGenerator(beta=1.5)
    matcher = Matcher()
    
    # Configuration
    data_dir = "../sample"  
    user_id = 1        
    threshold = 100.0   
    
    # Enrollment Phase - Create user template
    print(f"\nEnrolling user {user_id}...")
    
    # Get genuine signature files for this user
    enrolled_files = reader.get_user_files(data_dir, user_id, genuine=True)
    print(f"Found {len(enrolled_files)} genuine signatures for enrollment")
    
    # Process each signature and extract features
    enrolled_features = []
    for i, file in enumerate(enrolled_files[:20]):  
        print(f"Processing enrollment signature {i+1}...")
        X, Y, P = reader.read_signature(os.path.join(data_dir, file))
        features = extractor.process_signature(X, Y, P)
        enrolled_features.append(features)
    
    # Generate user template
    template, q = template_gen.generate_template(enrolled_features)
    print("User template generated successfully")
    
    # Verification Phase - Test with genuine and forged signatures
    print("\nTesting verification system...")
    
    # Test with the genuine signatures
    for i in range(0, 20):
        test_file = enrolled_files[i]  # Different from enrolled ones
        print(f"\nTesting with genuine signature: {test_file}")
        X, Y, P = reader.read_signature(os.path.join(data_dir, test_file))
        test_features = extractor.process_signature(X, Y, P)
        score = matcher.verify_signature(test_features, template, q)
        print(f"Dissimilarity score: {score:.2f}")
        print("Verification:", "Accepted" if score < threshold else "Rejected")
    
    # Test with the forged signatures
    forged_files = reader.get_user_files(data_dir, user_id, genuine=False)
    for i in range(0, 20):
        test_file = forged_files[i]
        print(f"\nTesting with forged signature: {test_file}")
        X, Y, P = reader.read_signature(os.path.join(data_dir, test_file))
        test_features = extractor.process_signature(X, Y, P)
        score = matcher.verify_signature(test_features, template, q)
        print(f"Dissimilarity score: {score:.2f}")
        print("Verification:", "Accepted" if score < threshold else "Rejected")

if __name__ == "__main__":
    main()
