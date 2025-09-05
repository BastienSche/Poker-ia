"""
Traitement et optimisation des images pour l'analyse de poker
"""
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional

class PokerImageProcessor:
    """Processeur d'images optimisé pour les tables de poker"""
    
    def __init__(self):
        # Résolution ULTRA-OPTIMISÉE pour vitesse maximale
        self.target_width = 800  # Réduit encore plus
        self.target_height = 450  # Réduit encore plus
        # Qualité JPEG réduite pour vitesse
        self.jpeg_quality = 60  # Réduit pour vitesse
        
    def optimize_image_for_poker_analysis(self, image_base64: str) -> str:
        """
        Optimise une image pour l'analyse ULTRA-RAPIDE :
        - Redimensionnement agressif
        - Compression rapide
        - Traitement minimal
        """
        try:
            # Décodage de l'image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Conversion en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionnement AGRESSIF pour vitesse
            image = self._fast_resize(image)
            
            # Amélioration LÉGÈRE seulement
            image = self._light_enhance(image)
            
            # Compression RAPIDE
            optimized_base64 = self._fast_compress(image)
            
            return optimized_base64
            
        except Exception as e:
            print(f"Erreur lors de l'optimisation d'image: {e}")
            return image_base64  # Retour de l'image originale en cas d'erreur
    
    def _fast_resize(self, image: Image.Image) -> Image.Image:
        """Redimensionnement ultra-rapide"""
        # Redimensionnement direct sans préservation parfaite des proportions pour vitesse
        return image.resize((self.target_width, self.target_height), Image.Resampling.BILINEAR)
    
    def _light_enhance(self, image: Image.Image) -> Image.Image:
        """Amélioration légère ultra-rapide"""
        # Seulement un léger contraste pour vitesse
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.1)  # Très léger
    
    def _fast_compress(self, image: Image.Image) -> str:
        """Compression ultra-rapide"""
        buffer = io.BytesIO()
        
        # Sauvegarde rapide sans optimisations coûteuses
        image.save(
            buffer, 
            format='JPEG', 
            quality=self.jpeg_quality
            # Pas d'optimize=True ni progressive=True pour plus de vitesse
        )
        
        # Encodage en base64
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def detect_poker_elements_regions(self, image_base64: str) -> dict:
        """
        Détecte les régions d'intérêt sur une table de poker
        (pour des optimisations futures)
        """
        # Cette fonction pourrait être étendue pour détecter automatiquement
        # les zones de cartes, de jetons, etc. pour un recadrage intelligent
        return {
            "full_table": True,
            "cards_region": None,
            "player_regions": [],
            "pot_region": None
        }