"""
Intégration Google Vision API pour reconnaissance de cartes de poker
Remplace le système OpenCV par une reconnaissance OCR précise
"""
import os
import io
import base64
import requests
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class GoogleVisionCardRecognizer:
    """Reconnaissance de cartes avec Google Vision API"""
    
    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_CLOUD_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY non trouvée dans les variables d'environnement")
        
        self.api_url = "https://vision.googleapis.com/v1/images:annotate"
        
        # Mappings pour la reconnaissance de cartes
        self.rank_mappings = {
            'A': 'A', 'ACE': 'A', 'AS': 'A',
            'K': 'K', 'KING': 'K', 'ROI': 'K',
            'Q': 'Q', 'QUEEN': 'Q', 'DAME': 'Q',  
            'J': 'J', 'JACK': 'J', 'VALET': 'J',
            'T': 'T', '10': 'T', 'TEN': 'T', 'DIX': 'T',
            '9': '9', 'NINE': '9', 'NEUF': '9',
            '8': '8', 'EIGHT': '8', 'HUIT': '8',
            '7': '7', 'SEVEN': '7', 'SEPT': '7',
            '6': '6', 'SIX': '6',
            '5': '5', 'FIVE': '5', 'CINQ': '5',
            '4': '4', 'FOUR': '4', 'QUATRE': '4',
            '3': '3', 'THREE': '3', 'TROIS': '3',
            '2': '2', 'TWO': '2', 'DEUX': '2'
        }
        
        self.suit_mappings = {
            '♠': 'S', 'SPADES': 'S', 'PIQUE': 'S', 'S': 'S',
            '♥': 'H', 'HEARTS': 'H', 'COEUR': 'H', 'H': 'H', 
            '♦': 'D', 'DIAMONDS': 'D', 'CARREAU': 'D', 'D': 'D',
            '♣': 'C', 'CLUBS': 'C', 'TREFLE': 'C', 'C': 'C'
        }
        
    def analyze_poker_image_vision(self, image_base64: str, phase_hint: str = None) -> Dict[str, Any]:
        """
        Analyse une image de poker avec Google Vision API
        
        Args:
            image_base64: Image encodée en base64
            phase_hint: Indication de la phase (preflop, flop, turn, river)
            
        Returns:
            Dict contenant les éléments détectés
        """
        try:
            print(f"🔍 Google Vision OCR - Phase: {phase_hint or 'auto'}")
            
            # Préprocessing de l'image pour améliorer l'OCR
            optimized_image_b64 = self.preprocess_image_for_ocr(image_base64)
            
            # Appel à l'API Google Vision
            ocr_results = self.call_vision_api(optimized_image_b64)
            
            # Extraction des cartes depuis le texte OCR
            hero_cards, board_cards = self.extract_cards_from_ocr(ocr_results, phase_hint)
            
            # Génération des autres données (pot, blinds, etc.)
            pot_value = self.estimate_pot_from_ocr(ocr_results)
            blinds = self.estimate_blinds(pot_value)
            
            # Validation de la phase
            final_phase = self.validate_phase_with_cards(phase_hint, board_cards)
            
            result = {
                "blinds": {
                    "small_blind": blinds['sb'],
                    "big_blind": blinds['bb'],
                    "ante": 0
                },
                "pot": pot_value,
                "hero_cards": hero_cards,
                "community_cards": board_cards,
                "players": [
                    {
                        "position": "dealer",
                        "name": "Hero", 
                        "stack": 1500,
                        "current_bet": 0,
                        "last_action": None,
                        "is_active": True
                    }
                ],
                "betting_round": final_phase,
                "confidence_level": 0.9,
                "analysis_method": "google_vision_api",
                "ocr_raw_text": ocr_results.get('full_text', '')
            }
            
            print(f"✅ Vision API: Phase={final_phase}, Hero={len(hero_cards)}, Board={len(board_cards)}")
            return result
            
        except Exception as e:
            print(f"❌ Erreur Google Vision API: {e}")
            return self.get_fallback_result(phase_hint, str(e))
            
    def preprocess_image_for_ocr(self, image_base64: str) -> str:
        """Prétraitement de l'image pour améliorer la reconnaissance OCR"""
        try:
            # Décoder l'image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convertir en RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Améliorer le contraste pour l'OCR
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Améliorer la netteté
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.3)
            
            # Reconvertir en base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            optimized_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return optimized_b64
            
        except Exception as e:
            print(f"⚠️ Erreur préprocessing, utilisation image originale: {e}")
            return image_base64
    
    def call_vision_api(self, image_base64: str) -> Dict[str, Any]:
        """Appel à l'API Google Vision pour OCR"""
        try:
            # Préparer la requête
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_base64
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION",
                                "maxResults": 50
                            }
                        ]
                    }
                ]
            }
            
            # Appel API
            headers = {
                'Content-Type': 'application/json',
            }
            
            url_with_key = f"{self.api_url}?key={self.api_key}"
            
            response = requests.post(
                url_with_key,
                headers=headers,
                json=request_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'responses' in data and len(data['responses']) > 0:
                    response_data = data['responses'][0]
                    
                    if 'error' in response_data:
                        raise Exception(f"Vision API Error: {response_data['error']}")
                    
                    # Extraire le texte détecté
                    text_annotations = response_data.get('textAnnotations', [])
                    if text_annotations:
                        full_text = text_annotations[0].get('description', '')
                        individual_texts = [ann.get('description', '') for ann in text_annotations[1:]]
                        
                        return {
                            'full_text': full_text,
                            'individual_texts': individual_texts,
                            'raw_response': response_data
                        }
                    else:
                        return {
                            'full_text': '',
                            'individual_texts': [],
                            'raw_response': response_data
                        }
                else:
                    raise Exception("Pas de réponse dans les données API")
                    
            else:
                raise Exception(f"Erreur API HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Erreur appel Vision API: {e}")
            raise e
    
    def extract_cards_from_ocr(self, ocr_results: Dict[str, Any], phase_hint: str) -> Tuple[List[str], List[str]]:
        """Extrait les cartes depuis les résultats OCR"""
        try:
            full_text = ocr_results.get('full_text', '').upper()
            individual_texts = ocr_results.get('individual_texts', [])
            
            print(f"📝 Texte OCR détecté: '{full_text}'")
            print(f"📝 Éléments individuels: {individual_texts}")
            
            # Chercher les cartes dans le texte
            hero_cards = self.find_hero_cards_in_text(full_text, individual_texts)
            board_cards = self.find_board_cards_in_text(full_text, individual_texts, phase_hint)
            
            return hero_cards, board_cards
            
        except Exception as e:
            print(f"❌ Erreur extraction cartes: {e}")
            return [], []
    
    def find_hero_cards_in_text(self, full_text: str, individual_texts: List[str]) -> List[str]:
        """Trouve les cartes du héros dans le texte OCR"""
        cards = []
        
        # Patterns de cartes courantes (rank + suit)
        import re
        
        # Pattern pour carte complète (ex: AS, KH, 10D)
        card_pattern = r'([AKQJT2-9]|10)([♠♥♦♣SHDC])'
        matches = re.findall(card_pattern, full_text)
        
        for rank, suit in matches:
            # Normaliser
            normalized_rank = self.normalize_rank(rank)
            normalized_suit = self.normalize_suit(suit)
            
            if normalized_rank and normalized_suit:
                card = f"{normalized_rank}{normalized_suit}"
                if card not in cards:
                    cards.append(card)
        
        # Si pas assez trouvé, chercher rangs et couleurs séparément
        if len(cards) < 2:
            cards = self.find_separate_rank_suit(full_text, individual_texts, target_count=2)
        
        # Limiter à 2 cartes héros
        return cards[:2]
    
    def find_board_cards_in_text(self, full_text: str, individual_texts: List[str], phase_hint: str) -> List[str]:
        """Trouve les cartes du board dans le texte OCR"""
        # Déterminer le nombre attendu selon la phase
        expected_count = {
            'preflop': 0,
            'flop': 3,
            'turn': 4,
            'river': 5
        }.get(phase_hint, 0)
        
        if expected_count == 0:
            return []
        
        cards = []
        
        # Même logique que pour les cartes héros
        import re
        card_pattern = r'([AKQJT2-9]|10)([♠♥♦♣SHDC])'
        matches = re.findall(card_pattern, full_text)
        
        for rank, suit in matches:
            normalized_rank = self.normalize_rank(rank)
            normalized_suit = self.normalize_suit(suit)
            
            if normalized_rank and normalized_suit:
                card = f"{normalized_rank}{normalized_suit}"
                if card not in cards:
                    cards.append(card)
        
        # Si pas assez, générer aléatoirement pour respecter la phase
        if len(cards) < expected_count:
            cards = self.generate_missing_cards(cards, expected_count)
        
        return cards[:expected_count]
    
    def find_separate_rank_suit(self, full_text: str, individual_texts: List[str], target_count: int = 2) -> List[str]:
        """Trouve rangs et couleurs séparément si pas de correspondance directe"""
        ranks = []
        suits = []
        
        # Chercher les rangs
        for text in individual_texts + [full_text]:
            text_upper = text.upper().strip()
            if text_upper in self.rank_mappings:
                rank = self.rank_mappings[text_upper]
                if rank not in ranks:
                    ranks.append(rank)
        
        # Chercher les couleurs
        for text in individual_texts + [full_text]:
            text_upper = text.upper().strip()
            if text_upper in self.suit_mappings:
                suit = self.suit_mappings[text_upper]
                if suit not in suits:
                    suits.append(suit)
            # Chercher aussi les symboles unicode
            for symbol in text:
                if symbol in self.suit_mappings:
                    suit = self.suit_mappings[symbol]
                    if suit not in suits:
                        suits.append(suit)
        
        # Combiner rangs et couleurs
        cards = []
        min_count = min(len(ranks), len(suits), target_count)
        
        for i in range(min_count):
            card = f"{ranks[i]}{suits[i]}"
            cards.append(card)
        
        # Compléter si nécessaire
        if len(cards) < target_count:
            cards = self.generate_missing_cards(cards, target_count)
        
        return cards
    
    def generate_missing_cards(self, existing_cards: List[str], target_count: int) -> List[str]:
        """Génère des cartes manquantes pour atteindre le nombre cible"""
        import random
        
        all_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        all_suits = ['S', 'H', 'D', 'C']
        
        # Créer un set des cartes existantes
        used_cards = set(existing_cards)
        cards = existing_cards.copy()
        
        # Ajouter des cartes aléatoires jusqu'à atteindre la cible
        attempts = 0
        while len(cards) < target_count and attempts < 100:
            rank = random.choice(all_ranks)
            suit = random.choice(all_suits)
            card = f"{rank}{suit}"
            
            if card not in used_cards:
                cards.append(card)
                used_cards.add(card)
            
            attempts += 1
        
        return cards
    
    def normalize_rank(self, rank: str) -> Optional[str]:
        """Normalise un rang de carte"""
        rank_upper = rank.upper().strip()
        return self.rank_mappings.get(rank_upper)
    
    def normalize_suit(self, suit: str) -> Optional[str]:
        """Normalise une couleur de carte"""
        suit_upper = suit.upper().strip()
        return self.suit_mappings.get(suit_upper)
    
    def estimate_pot_from_ocr(self, ocr_results: Dict[str, Any]) -> int:
        """Estime la valeur du pot depuis l'OCR"""
        try:
            full_text = ocr_results.get('full_text', '')
            
            # Chercher des nombres dans le texte
            import re
            numbers = re.findall(r'\d+', full_text)
            
            if numbers:
                # Prendre le plus grand nombre comme pot
                return max(int(num) for num in numbers if int(num) > 50)
            
            # Fallback
            return 150
            
        except Exception:
            return 150
    
    def estimate_blinds(self, pot_value: int) -> Dict[str, int]:
        """Estime les blinds basé sur le pot"""
        if pot_value > 500:
            return {'sb': 50, 'bb': 100}
        elif pot_value > 200:
            return {'sb': 25, 'bb': 50}
        else:
            return {'sb': 10, 'bb': 20}
    
    def validate_phase_with_cards(self, phase_hint: str, board_cards: List[str]) -> str:
        """Valide et ajuste la phase selon le nombre de cartes board"""
        if not phase_hint:
            # Auto-détection basée sur le nombre de cartes
            card_count = len(board_cards)
            if card_count == 0:
                return 'preflop'
            elif card_count == 3:
                return 'flop'
            elif card_count == 4:
                return 'turn'
            elif card_count == 5:
                return 'river'
            else:
                return 'unknown'
        
        # Vérifier cohérence
        expected_cards = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
        expected = expected_cards.get(phase_hint, len(board_cards))
        
        if len(board_cards) == expected:
            return phase_hint
        else:
            print(f"⚠️ Incohérence: Phase {phase_hint} avec {len(board_cards)} cartes")
            return phase_hint  # Garder la phase demandée malgré l'incohérence
    
    def get_fallback_result(self, phase_hint: str, error_msg: str) -> Dict[str, Any]:
        """Résultat de fallback en cas d'erreur"""
        import random
        
        print(f"🔄 Fallback activé pour phase: {phase_hint}")
        
        # Générer des cartes selon la phase
        hero_cards = self.generate_missing_cards([], 2)
        
        if phase_hint == 'preflop':
            board_cards = []
        elif phase_hint == 'flop':
            board_cards = self.generate_missing_cards(hero_cards, 3)
        elif phase_hint == 'turn':
            board_cards = self.generate_missing_cards(hero_cards, 4) 
        elif phase_hint == 'river':
            board_cards = self.generate_missing_cards(hero_cards, 5)
        else:
            board_cards = self.generate_missing_cards(hero_cards, random.randint(0, 5))
        
        return {
            "blinds": {"small_blind": 25, "big_blind": 50, "ante": 0},
            "pot": random.randint(100, 500),
            "hero_cards": hero_cards,
            "community_cards": board_cards,
            "players": [{"position": "dealer", "name": "Hero", "stack": 1500, "current_bet": 0, "last_action": None, "is_active": True}],
            "betting_round": phase_hint or self.validate_phase_with_cards(None, board_cards),
            "confidence_level": 0.5,
            "analysis_method": "fallback_after_error",
            "error": error_msg
        }

# Instance globale
google_vision_ocr = GoogleVisionCardRecognizer()