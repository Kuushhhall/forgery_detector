"""
CLIP-based probe for synthetic image detection.

This module uses CLIP to compute semantic similarity between images
and text prompts indicating natural vs. synthetic origin.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Optional
import cv2


class CLIPProbe:
    """
    Wrapper for CLIP-based synthetic detection.
    Can be extended with a linear probe trained on labeled data.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model and processor.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Optional: Load trained linear probe
        self.linear_probe = None  # Placeholder for sklearn classifier
    
    def compute_similarity(self, image: np.ndarray, prompts: list[str]) -> np.ndarray:
        """
        Compute CLIP similarity scores between image and text prompts.
        
        Args:
            image: Input image (BGR format from OpenCV)
            prompts: List of text descriptions
            
        Returns:
            Array of similarity probabilities (sums to 1)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(image)
        
        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=pil_img,
            return_tensors="pt",
            padding=True
        )
        
        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        return probs
    
    def detect_synthetic(self, image: np.ndarray) -> float:
        """
        Detect if image is likely AI-generated or synthetic.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Score 0-1, where higher = more likely synthetic
        """
        prompts = [
            "a natural photograph taken with a camera",
            "a real photo of the scene",
            "an AI generated image",
            "a computer rendered image",
            "a synthetic digital creation"
        ]
        
        probs = self.compute_similarity(image, prompts)
        
        # Natural score (first two prompts)
        natural_score = probs[0] + probs[1]
        
        # Synthetic score (last three prompts)
        synthetic_score = probs[2] + probs[3] + probs[4]
        
        # Normalize to [0, 1]
        total = natural_score + synthetic_score
        if total > 0:
            synthetic_score = synthetic_score / total
        
        return float(np.clip(synthetic_score, 0, 1))
    
    def train_linear_probe(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train a simple linear classifier on CLIP embeddings.
        
        Example usage:
            probe = CLIPProbe()
            
            # Compute embeddings for training images
            embeddings = []
            for img in train_images:
                emb = probe.get_image_embedding(img)
                embeddings.append(emb)
            
            X_train = np.array(embeddings)
            y_train = np.array([0, 0, 1, 1, ...])  # 0=real, 1=fake
            
            probe.train_linear_probe(X_train, y_train)
        
        Args:
            X_train: Array of CLIP embeddings (N, 512)
            y_train: Binary labels (N,) where 1=synthetic, 0=real
        """
        from sklearn.linear_model import LogisticRegression
        
        self.linear_probe = LogisticRegression(max_iter=1000)
        self.linear_probe.fit(X_train, y_train)
        print(f"Linear probe trained on {len(X_train)} samples")
    
    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract CLIP image embedding.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Embedding vector (512,)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(image)
        
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.cpu().numpy()[0]
        
        return embedding
    
    def predict_with_probe(self, image: np.ndarray) -> float:
        """
        Predict using trained linear probe (if available).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Probability of being synthetic [0, 1]
        """
        if self.linear_probe is None:
            raise ValueError("Linear probe not trained. Call train_linear_probe() first.")
        
        embedding = self.get_image_embedding(image).reshape(1, -1)
        prob = self.linear_probe.predict_proba(embedding)[0, 1]
        
        return float(prob)


# Example usage
if __name__ == "__main__":
    # Initialize probe
    probe = CLIPProbe()
    
    # Test with sample image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Zero-shot detection
    score = probe.detect_synthetic(test_image)
    print(f"Synthetic score: {score:.3f}")
    
    # Example: Train linear probe on small dataset
    # (You would replace this with real training data)
    n_samples = 100
    dummy_embeddings = np.random.randn(n_samples, 512)
    dummy_labels = np.random.randint(0, 2, n_samples)
    
    probe.train_linear_probe(dummy_embeddings, dummy_labels)
    
    # Now predict with trained probe
    # score_trained = probe.predict_with_probe(test_image)
    # print(f"Trained probe score: {score_trained:.3f}")
