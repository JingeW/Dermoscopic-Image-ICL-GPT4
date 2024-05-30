"""
Find the top-k similar images to the query image.
Use VIT ImageNet pretrained model to extract the feature.
Compute Cosine similarity to define the feature-level similar images.
"""
import os
import json
import argparse
import torch
from PIL import Image, UnidentifiedImageError
from transformers import ViTModel, ViTFeatureExtractor, ViTConfig
from scipy.spatial.distance import cosine

class FeatureExtractor:
    def __init__(self, model_name='vit_base_patch16_224'):
        self.model = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0]  # Extract the class token
            return features.squeeze().detach().numpy()
        except UnidentifiedImageError:
            print(f"Could not open image: {img_path}")
            return None

def extract_features_from_directory(image_dir, feature_extractor):
    image_features = {}
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_name)
            features = feature_extractor.extract_features(img_path)
            if features is not None:
                image_features[img_name] = features
    return image_features

def compute_top_k_similarities(query_features, example_img_features, k):
    similarity_matrix = {}
    for query_name, query_vec in query_features.items():
        similarities = [(name, 1 - cosine(query_vec, example_img_features[name]))
                        for name in example_img_features if name != query_name]
        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        similarity_matrix[query_name] = top_k
    return similarity_matrix

def save_similarity_matrix(similarity_matrix, file_path):
    with open(file_path, 'w') as file:
        json.dump(similarity_matrix, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute top-k similar images using VIT and cosine similarity.")
    parser.add_argument('--query_dir', type=str, default='./data/all_resized', help='Directory of query images.')
    parser.add_argument('--example_dir', type=str, default='./data/mm_resized', help='Directory of example images.')
    parser.add_argument('--output_file', type=str, default='./data/mm_similarity_matrix_vit.json', help='Output file path for similarity matrix.')
    parser.add_argument('--k', type=int, default=10, help='Number of top similar images to find.')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224', help='Model name for VIT.')
    args = parser.parse_args()

    feature_extractor = FeatureExtractor(model_name=args.model_name)
    query_features = extract_features_from_directory(args.query_dir, feature_extractor)
    example_img_features = extract_features_from_directory(args.example_dir, feature_extractor)
    similarities = compute_top_k_similarities(query_features, example_img_features, args.k)
    save_similarity_matrix(similarities, args.output_file)
