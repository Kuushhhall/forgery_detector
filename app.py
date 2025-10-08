"""
Image Forgery Detection - Streamlit MVP
A production-oriented prototype for detecting image tampering and AI-generated images.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import io
import json
from datetime import datetime
import exifread
from typing import Dict, Tuple, Optional
import base64

# Page config
st.set_page_config(
    page_title="Image Forgery Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional forensic interface
st.markdown("""
<style>
    .main {
        background-color: #0f1419;
    }
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    h1, h2, h3 {
        color: #00d4ff;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .score-bar {
        height: 24px;
        border-radius: 4px;
        background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);
        margin: 8px 0;
    }
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 3px solid #00d4ff;
        padding: 12px;
        margin: 12px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Scoring weights (configurable)
DEFAULT_WEIGHTS = {
    "clip_probe": 0.30,
    "cnn_classifier": 0.35,
    "frequency_analysis": 0.15,
    "texture_residual": 0.10,
    "copy_move": 0.05,
    "metadata": 0.05
}


def load_image(uploaded_file) -> Optional[np.ndarray]:
    """Load image with robust fallback handling."""
    try:
        # Try PIL first
        pil_img = Image.open(uploaded_file).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None


def extract_exif(uploaded_file) -> Dict:
    """Extract EXIF metadata and flag suspicious patterns."""
    uploaded_file.seek(0)
    tags = exifread.process_file(uploaded_file, details=False)
    
    exif_data = {str(key): str(tags[key]) for key in tags.keys()}
    
    # Heuristic flags
    flags = {
        "missing_exif": len(exif_data) == 0,
        "missing_gps": not any('GPS' in k for k in exif_data.keys()),
        "missing_datetime": not any('DateTime' in k for k in exif_data.keys()),
        "suspicious_software": any(
            sw in str(exif_data.get('Image Software', '')).lower() 
            for sw in ['photoshop', 'gimp', 'paint', 'stable diffusion', 'midjourney']
        )
    }
    
    return {"data": exif_data, "flags": flags}


def compute_ela(image: np.ndarray, quality: int = 90) -> np.ndarray:
    """Error Level Analysis - detect compression artifacts."""
    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Compress and decompress
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    # Compute difference
    ela = ImageChops.difference(pil_img, compressed)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return cv2.cvtColor(np.array(ela), cv2.COLOR_RGB2BGR)


def compute_fft(image: np.ndarray) -> np.ndarray:
    """Frequency domain analysis using FFT."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)


def compute_noise_residual(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Extract noise residual (PRNU proxy) and compute score."""
    # Denoise using Gaussian blur
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Compute residual
    residual = cv2.subtract(image, denoised)
    
    # Compute uniformity score (lower = more suspicious)
    gray_residual = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_residual)
    mean_val = np.mean(gray_residual)
    
    # Normalize score (higher std_dev relative to mean = more natural)
    score = min(1.0, std_dev / (mean_val + 1e-6) / 10.0)
    
    return residual, score


def detect_copy_move(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Copy-move detection using ORB + BFMatcher."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)
    
    if des is None or len(kp) < 10:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des, des, k=2)
    
    # Filter good matches (similar but not identical)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance and m.queryIdx != m.trainIdx:
                # Check spatial distance
                pt1 = kp[m.queryIdx].pt
                pt2 = kp[m.trainIdx].pt
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if 20 < dist < 200:  # Not too close, not too far
                    good_matches.append(m)
    
    # Create mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for m in good_matches:
        pt1 = tuple(map(int, kp[m.queryIdx].pt))
        pt2 = tuple(map(int, kp[m.trainIdx].pt))
        cv2.circle(mask, pt1, 10, 255, -1)
        cv2.circle(mask, pt2, 10, 255, -1)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Score based on number of suspicious matches
    score = min(1.0, len(good_matches) / 50.0)
    
    return mask, score


def clip_probe_score(image: np.ndarray) -> float:
    """
    CLIP-based probe for synthetic image detection.
    Returns normalized score (0-1) for "likely synthetic".
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        # Load CLIP model (cache this in production)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Convert image
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare prompts
        prompts = [
            "a natural photograph",
            "an AI generated image",
            "a computer rendered image",
            "a real photograph taken with a camera"
        ]
        
        inputs = processor(text=prompts, images=pil_img, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Synthetic score = (AI generated + rendered) / (natural + real)
        synthetic_score = (probs[1] + probs[2]) / (probs[0] + probs[3] + 1e-6)
        return float(np.clip(synthetic_score, 0, 1))
        
    except Exception as e:
        st.warning(f"CLIP probe failed: {e}. Using fallback.")
        return 0.5  # Neutral fallback


def cnn_classify_and_gradcam(image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """
    Placeholder for CNN classification + Grad-CAM.
    Drop your trained model here and return (probability, gradcam_overlay).
    """
    # TODO: Load your CNN model (.pt or .h5)
    # model = torch.load('models/cnn_forgery.pt')
    # prob = model(preprocess(image))
    # gradcam = generate_gradcam(model, image)
    
    # Placeholder: return dummy values
    return 0.5, None


def maskrcnn_localize(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Placeholder for Mask R-CNN localization.
    Return mask overlay showing detected manipulated regions.
    """
    # TODO: Load Mask R-CNN and run inference
    # masks = maskrcnn_model(image)
    # composite_mask = combine_masks(masks)
    
    return None


def compute_final_score(module_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted fusion of module scores."""
    score = 0.0
    for module, weight in weights.items():
        score += module_scores.get(module, 0.5) * weight
    return np.clip(score, 0, 1)


def interpret_score(score: float) -> Tuple[str, str]:
    """Return label and color for confidence score."""
    if score < 0.35:
        return "Likely Authentic", "#10b981"
    elif score < 0.6:
        return "Suspicious", "#f59e0b"
    else:
        return "Likely Forgery/AI", "#ef4444"


def main():
    st.title("🔍 Image Forgery Detector")
    st.markdown("**Professional forensic analysis for image authenticity verification**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Detection Weights")
        weights = {}
        for module, default_weight in DEFAULT_WEIGHTS.items():
            weights[module] = st.slider(
                module.replace('_', ' ').title(),
                0.0, 1.0, default_weight, 0.05
            )
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        st.markdown("---")
        st.markdown("""
        ### 📊 How to Interpret
        
        - **0.0 - 0.35**: Likely authentic
        - **0.35 - 0.6**: Suspicious, needs review
        - **0.6 - 1.0**: Likely AI-generated or tampered
        
        ### ⚠️ Limitations
        
        - Detectors are probabilistic, not definitive
        - EXIF metadata can be spoofed
        - Heavy compression reduces accuracy
        - Use as triage, not legal proof
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload image for forensic analysis"
        )
        
        if uploaded_file:
            # Display original
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
    
    with col2:
        if uploaded_file:
            st.subheader("🔬 Analysis Results")
            
            with st.spinner("Running forensic analysis..."):
                # Load image
                image = load_image(uploaded_file)
                if image is None:
                    st.error("Failed to load image")
                    return
                
                # Run all analyses
                module_scores = {}
                
                # 1. EXIF Analysis
                with st.expander("📋 EXIF Metadata", expanded=False):
                    exif_result = extract_exif(uploaded_file)
                    flags = exif_result['flags']
                    
                    flag_score = sum([
                        flags['missing_exif'] * 0.3,
                        flags['missing_gps'] * 0.1,
                        flags['missing_datetime'] * 0.2,
                        flags['suspicious_software'] * 0.4
                    ])
                    module_scores['metadata'] = flag_score
                    
                    st.json(flags)
                    if exif_result['data']:
                        st.json(dict(list(exif_result['data'].items())[:10]))
                
                # 2. ELA
                with st.expander("🔍 Error Level Analysis (ELA)", expanded=True):
                    ela = compute_ela(image)
                    st.image(ela, caption="ELA Visualization", use_column_width=True, channels="BGR")
                    st.caption("Bright areas indicate different compression levels (potential editing)")
                
                # 3. FFT
                with st.expander("📊 Frequency Analysis (FFT)", expanded=True):
                    fft = compute_fft(image)
                    st.image(fft, caption="FFT Magnitude Spectrum", use_column_width=True, channels="BGR")
                    
                    # Score based on symmetry
                    gray_fft = cv2.cvtColor(fft, cv2.COLOR_BGR2GRAY)
                    h, w = gray_fft.shape
                    left = gray_fft[:, :w//2]
                    right = cv2.flip(gray_fft[:, w//2:], 1)
                    diff = np.mean(np.abs(left - right[:, :left.shape[1]]))
                    freq_score = min(1.0, diff / 50.0)
                    module_scores['frequency_analysis'] = freq_score
                
                # 4. Noise Residual
                with st.expander("🌊 Noise Residual (PRNU Proxy)", expanded=True):
                    residual, texture_score = compute_noise_residual(image)
                    st.image(residual, caption="Noise Pattern", use_column_width=True, channels="BGR")
                    module_scores['texture_residual'] = 1.0 - texture_score  # Invert: low uniformity = suspicious
                    st.caption(f"Uniformity score: {texture_score:.3f} (lower = more suspicious)")
                
                # 5. Copy-Move Detection
                with st.expander("📍 Copy-Move Detection", expanded=True):
                    mask, cm_score = detect_copy_move(image)
                    
                    # Overlay mask on original
                    overlay = image.copy()
                    overlay[mask > 0] = [0, 0, 255]  # Red highlight
                    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                    
                    st.image(result, caption="Copy-Move Mask Overlay", use_column_width=True, channels="BGR")
                    module_scores['copy_move'] = cm_score
                    st.caption(f"Suspicious regions detected: {cm_score:.3f}")
                
                # 6. CLIP Probe
                with st.expander("🤖 AI Generation Probe (CLIP)", expanded=True):
                    clip_score = clip_probe_score(image)
                    module_scores['clip_probe'] = clip_score
                    st.metric("Synthetic Likelihood", f"{clip_score:.3f}")
                    st.caption("CLIP-based semantic analysis for AI-generated content")
                
                # 7. CNN Classifier (Placeholder)
                cnn_prob, gradcam = cnn_classify_and_gradcam(image)
                module_scores['cnn_classifier'] = cnn_prob
                
                # Compute final score
                final_score = compute_final_score(module_scores, weights)
                label, color = interpret_score(final_score)
                
                # Display final verdict
                st.markdown("---")
                st.markdown(f"### 🎯 Final Confidence: **{final_score:.3f}**")
                st.markdown(f"<div style='background:{color};padding:12px;border-radius:8px;text-align:center;'>"
                           f"<h2 style='color:white;margin:0;'>{label}</h2></div>", 
                           unsafe_allow_html=True)
                
                # Module breakdown
                st.markdown("#### 📊 Module Scores")
                for module, score in module_scores.items():
                    st.progress(score, text=f"{module.replace('_', ' ').title()}: {score:.3f}")
                
                # Generate JSON report
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "final_score": float(final_score),
                    "verdict": label,
                    "module_scores": {k: float(v) for k, v in module_scores.items()},
                    "weights_used": weights,
                    "exif_flags": exif_result['flags'],
                    "image_shape": image.shape
                }
                
                st.download_button(
                    "📥 Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
