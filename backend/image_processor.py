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
    
    def _smart_resize(self, image: Image.Image) -> Image.Image:
        """Redimensionnement intelligent pour préserver les détails importants"""
        original_width, original_height = image.size
        
        # Calcul du ratio pour conserver les proportions
        width_ratio = self.target_width / original_width
        height_ratio = self.target_height / original_height
        
        # Utilisation du ratio le plus petit pour éviter la déformation
        ratio = min(width_ratio, height_ratio)
        
        # Nouvelles dimensions
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Redimensionnement avec algorithme de haute qualité
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _enhance_for_cards(self, image: Image.Image) -> Image.Image:
        """Améliore l'image spécifiquement pour la reconnaissance de cartes"""
        # Amélioration du contraste pour faire ressortir les cartes
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # +20% de contraste
        
        # Amélioration de la netteté pour les textes et symboles
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)  # +30% de netteté
        
        # Légère amélioration de la luminosité pour les cartes sombres
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)  # +10% de luminosité
        
        return image
    
    def _compress_image(self, image: Image.Image) -> str:
        """Compression JPEG optimisée"""
        buffer = io.BytesIO()
        
        # Sauvegarde avec optimisation JPEG
        image.save(
            buffer, 
            format='JPEG', 
            quality=self.jpeg_quality,
            optimize=True,  # Optimisation automatique
            progressive=True  # JPEG progressif pour un chargement plus rapide
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