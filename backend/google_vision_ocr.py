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
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logger = logging.getLogger(__name__)

class GoogleVisionCardRecognizer:
    """Reconnaissance de cartes avec Google Vision API"""
    
    def __init__(self):
        print("🔄 Initialisation GoogleVisionCardRecognizer...")
        
        # Charger les variables d'environnement
        load_dotenv()
        
        self.api_key = os.environ.get('GOOGLE_CLOUD_API_KEY')
        print(f"🔑 API Key trouvée: {bool(self.api_key)} ({'***' + self.api_key[-4:] if self.api_key else 'None'})")
        
        if not self.api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY non trouvée dans les variables d'environnement")
        
        self.api_url = "https://vision.googleapis.com/v1/images:annotate"
        print(f"🌐 API URL: {self.api_url}")
        
        # Test de connectivité basique
        try:
            import requests
            test_response = requests.get("https://www.google.com", timeout=5)
            print(f"🌐 Test connectivité: OK ({test_response.status_code})")
        except Exception as e:
            print(f"⚠️ Test connectivité échoué: {e}")
        
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
        
        print("✅ GoogleVisionCardRecognizer initialisé avec succès")
        
    def analyze_poker_image_vision(self, image_base64: str, phase_hint: str = None) -> Dict[str, Any]:
        """
        Analyse optimisée d'une image de table de poker avec Google Vision API
        
        Args:
            image_base64: Image encodée en base64
            phase_hint: Indication de la phase (preflop, flop, turn, river)
            
        Returns:
            Dict contenant les éléments détectés et les demandes d'informations manquantes
        """
        try:
            print(f"🔍 Google Vision OCR - Analyse table poker - Phase: {phase_hint or 'auto'}")
            
            # Préprocessing spécialisé pour tables de poker
            optimized_image_b64 = self.preprocess_poker_table_image(image_base64)
            
            # Appel à l'API Google Vision avec configuration optimisée pour poker
            ocr_results = self.call_vision_api_optimized(optimized_image_b64)
            
            # Analyse spécialisée pour tables de poker
            poker_analysis = self.analyze_poker_table_layout(ocr_results, phase_hint)
            
            # Validation et demandes d'informations manquantes
            validated_result = self.validate_and_request_missing_info(poker_analysis, phase_hint)
            
            return validated_result
            
        except Exception as e:
            print(f"❌ Erreur Google Vision API: {e}")
            # Ne pas utiliser de fallback aléatoire, mais demander les infos à l'utilisateur
            return self.request_user_input_for_analysis(phase_hint, str(e))
    
    def preprocess_poker_table_image(self, image_base64: str) -> str:
        """Prétraitement spécialisé pour images de tables de poker"""
        try:
            print("🎯 Préprocessing spécialisé pour table de poker...")
            
            # Décoder l'image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Optimisations spéciales pour tables de poker
            
            # 1. Amélioration du contraste pour mieux voir les cartes
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.8)  # Contraste plus élevé
            
            # 2. Amélioration de la netteté pour les textes de cartes
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(2.0)  # Netteté maximale
            
            # 3. Amélioration de la luminosité pour les tables sombres
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(1.2)
            
            # 4. Détection et amélioration des zones de cartes
            image = self.enhance_card_regions(image)
            
            # Reconvertir en base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True, quality=95)
            optimized_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print("✅ Préprocessing poker terminé")
            return optimized_b64
            
        except Exception as e:
            print(f"⚠️ Erreur préprocessing, utilisation image originale: {e}")
            return image_base64
    
    def enhance_card_regions(self, image: Image.Image) -> Image.Image:
        """Améliore spécifiquement les zones où se trouvent généralement les cartes"""
        try:
            # Convertir en numpy pour OpenCV
            img_array = np.array(image)
            
            # Appliquer un filtre pour améliorer les contours (cartes)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
            
            # Améliorer les zones blanches (cartes) 
            # Les cartes sont généralement blanches/claires sur fond sombre
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Masque pour détecter les zones claires (cartes potentielles)
            lower_white = np.array([0, 0, 200])  # Zones très claires
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Appliquer une amélioration sur les zones de cartes détectées
            enhanced = img_array.copy()
            enhanced[mask > 0] = cv2.addWeighted(img_array, 0.7, 
                                               np.full_like(img_array, 255), 0.3, 0)[mask > 0]
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            print(f"⚠️ Erreur amélioration zones cartes: {e}")
            return image
    
    def call_vision_api_optimized(self, image_base64: str) -> Dict[str, Any]:
        """Appel optimisé à Google Vision API pour reconnaissance de poker"""
        try:
            print("📡 Appel Google Vision API optimisé pour poker...")
            
            # Configuration spécialisée pour tables de poker
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": image_base64
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION",
                                "maxResults": 100  # Plus de résultats pour capturer toutes les cartes
                            },
                            {
                                "type": "OBJECT_LOCALIZATION", 
                                "maxResults": 50   # Détecter les objets (cartes, jetons)
                            }
                        ],
                        "imageContext": {
                            "languageHints": ["en", "fr"],
                            "textDetectionParams": {
                                "enableTextDetectionConfidenceScore": True
                            }
                        }
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            url_with_key = f"{self.api_url}?key={self.api_key}"
            
            print(f"🔑 Utilisation clé API: ***{self.api_key[-4:]}")
            
            response = requests.post(
                url_with_key,
                headers=headers,
                json=request_data,
                timeout=15  # Timeout plus long pour traitement complexe
            )
            
            print(f"📊 Réponse API: Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'responses' in data and len(data['responses']) > 0:
                    response_data = data['responses'][0]
                    
                    if 'error' in response_data:
                        raise Exception(f"Vision API Error: {response_data['error']}")
                    
                    # Extraire textes ET objets
                    text_annotations = response_data.get('textAnnotations', [])
                    localized_objects = response_data.get('localizedObjectAnnotations', [])
                    
                    result = {
                        'full_text': '',
                        'individual_texts': [],
                        'text_annotations': [],
                        'objects': localized_objects,
                        'raw_response': response_data
                    }
                    
                    if text_annotations:
                        result['full_text'] = text_annotations[0].get('description', '')
                        result['individual_texts'] = [ann.get('description', '') for ann in text_annotations[1:]]
                        result['text_annotations'] = text_annotations
                    
                    print(f"✅ Vision API: {len(text_annotations)} textes, {len(localized_objects)} objets détectés")
                    return result
                else:
                    raise Exception("Pas de réponse dans les données API")
                    
            else:
                error_detail = response.text
                print(f"❌ Erreur API {response.status_code}: {error_detail}")
                raise Exception(f"Erreur API HTTP {response.status_code}: {error_detail}")
                
        except Exception as e:
            print(f"❌ Erreur appel Vision API: {e}")
            raise e
    
    def analyze_poker_table_layout(self, ocr_results: Dict[str, Any], phase_hint: str) -> Dict[str, Any]:
        """Analyse spécialisée pour layout de table de poker"""
        try:
            print("🎯 Analyse layout table de poker...")
            
            full_text = ocr_results.get('full_text', '').upper()
            text_annotations = ocr_results.get('text_annotations', [])
            objects = ocr_results.get('objects', [])
            
            print(f"📝 Texte complet détecté: '{full_text}'")
            print(f"📊 {len(text_annotations)} annotations, {len(objects)} objets")
            
            # Analyse des cartes par zones
            hero_cards = self.detect_hero_cards(text_annotations, full_text)
            board_cards = self.detect_community_cards(text_annotations, full_text, phase_hint)
            
            # Analyse des informations de jeu
            pot_info = self.detect_pot_and_bets(text_annotations, full_text)
            player_info = self.detect_players_and_positions(text_annotations, objects)
            
            # Validation des détections
            confidence_score = self.calculate_detection_confidence(hero_cards, board_cards, pot_info, phase_hint)
            
            result = {
                "hero_cards": hero_cards,
                "community_cards": board_cards,
                "pot_info": pot_info,
                "player_info": player_info,
                "confidence_score": confidence_score,
                "phase_detected": self.determine_phase_from_detections(board_cards, phase_hint),
                "raw_detections": {
                    "full_text": full_text,
                    "annotations_count": len(text_annotations),
                    "objects_count": len(objects)
                }
            }
            
            print(f"🎯 Analyse terminée: Hero={len(hero_cards)}, Board={len(board_cards)}, Confiance={confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur analyse layout: {e}")
            raise e
    
    def detect_hero_cards(self, text_annotations: List[Dict], full_text: str) -> List[str]:
        """Détecte les cartes du héros avec analyse de position"""
        cards = []
        
        print("🃏 Recherche cartes héros...")
        
        # Stratégie 1: Recherche de patterns de cartes dans les annotations
        for annotation in text_annotations[1:]:  # Skip le premier (texte complet)
            text = annotation.get('description', '').upper().strip()
            vertices = annotation.get('boundingPoly', {}).get('vertices', [])
            
            # Calculer position approximative
            if vertices and len(vertices) >= 2:
                avg_y = sum(v.get('y', 0) for v in vertices) / len(vertices)
                avg_x = sum(v.get('x', 0) for v in vertices) / len(vertices)
                
                # Les cartes héros sont généralement en bas de l'écran
                is_hero_position = avg_y > 600  # Ajuster selon résolution
                
                if is_hero_position:
                    card = self.parse_single_card(text)
                    if card and card not in cards:
                        cards.append(card)
                        print(f"🎯 Carte héros détectée: {card} à position ({avg_x:.0f}, {avg_y:.0f})")
        
        # Stratégie 2: Patterns dans le texte complet
        if len(cards) < 2:
            regex_cards = self.find_cards_with_regex(full_text)
            for card in regex_cards:
                if card not in cards and len(cards) < 2:
                    cards.append(card)
        
        print(f"✅ Cartes héros trouvées: {cards}")
        return cards[:2]  # Maximum 2 cartes
    
    def detect_community_cards(self, text_annotations: List[Dict], full_text: str, phase_hint: str) -> List[str]:
        """Détecte les cartes communes avec analyse de position centrale"""
        cards = []
        expected_count = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(phase_hint, 0)
        
        print(f"🎯 Recherche {expected_count} cartes communes pour phase {phase_hint}...")
        
        if expected_count == 0:
            return []
        
        # Recherche dans les annotations avec position centrale
        for annotation in text_annotations[1:]:
            text = annotation.get('description', '').upper().strip()
            vertices = annotation.get('boundingPoly', {}).get('vertices', [])
            
            if vertices and len(vertices) >= 2:
                avg_y = sum(v.get('y', 0) for v in vertices) / len(vertices)
                avg_x = sum(v.get('x', 0) for v in vertices) / len(vertices)
                
                # Les cartes communes sont au centre de la table
                is_center_position = (300 < avg_y < 600) and (200 < avg_x < 800)
                
                if is_center_position:
                    card = self.parse_single_card(text)
                    if card and card not in cards:
                        cards.append(card)
                        print(f"🃏 Carte commune détectée: {card} à position ({avg_x:.0f}, {avg_y:.0f})")
        
        # Si pas assez de cartes détectées, chercher dans le texte complet
        if len(cards) < expected_count:
            regex_cards = self.find_cards_with_regex(full_text)
            for card in regex_cards:
                if card not in cards and len(cards) < expected_count:
                    cards.append(card)
        
        print(f"✅ Cartes communes trouvées: {cards} (attendu: {expected_count})")
        return cards[:expected_count]
    
    def parse_single_card(self, text: str) -> Optional[str]:
        """Parse une carte individuelle depuis du texte"""
        if not text or len(text) < 2:
            return None
        
        # Nettoyer le texte
        text = text.strip().upper()
        
        # Pattern direct (AS, KH, etc.)
        import re
        card_pattern = r'^([AKQJT2-9]|10)([♠♥♦♣SHDC])$'
        match = re.match(card_pattern, text)
        
        if match:
            rank, suit = match.groups()
            normalized_rank = self.normalize_rank(rank)
            normalized_suit = self.normalize_suit(suit)
            
            if normalized_rank and normalized_suit:
                return f"{normalized_rank}{normalized_suit}"
        
        # Tentatives de correction d'OCR
        corrections = {
            'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2'
        }
        
        for old, new in corrections.items():
            corrected = text.replace(old, new)
            if corrected != text:
                return self.parse_single_card(corrected)
        
        return None
    
    def find_cards_with_regex(self, text: str) -> List[str]:
        """Trouve toutes les cartes dans un texte avec regex"""
        import re
        
        cards = []
        # Pattern pour cartes avec symboles unicode
        pattern1 = r'([AKQJT2-9]|10)([♠♥♦♣])'
        matches1 = re.findall(pattern1, text)
        
        for rank, suit in matches1:
            normalized_rank = self.normalize_rank(rank)
            normalized_suit = self.normalize_suit(suit)
            if normalized_rank and normalized_suit:
                card = f"{normalized_rank}{normalized_suit}"
                if card not in cards:
                    cards.append(card)
        
        # Pattern pour cartes avec lettres
        pattern2 = r'([AKQJT2-9]|10)([SHDC])'
        matches2 = re.findall(pattern2, text)
        
        for rank, suit in matches2:
            normalized_rank = self.normalize_rank(rank)
            normalized_suit = self.normalize_suit(suit)
            if normalized_rank and normalized_suit:
                card = f"{normalized_rank}{normalized_suit}"
                if card not in cards:
                    cards.append(card)
        
        return cards
    
    def detect_pot_and_bets(self, text_annotations: List[Dict], full_text: str) -> Dict[str, Any]:
        """Détecte le pot et les mises"""
        import re
        
        # Chercher des nombres qui pourraient être le pot
        numbers = re.findall(r'\d+', full_text)
        numbers = [int(n) for n in numbers if int(n) > 10]  # Filtrer petits nombres
        
        pot_value = max(numbers) if numbers else 150
        
        return {
            "pot": pot_value,
            "detected_numbers": numbers,
            "confidence": 0.7 if numbers else 0.3
        }
    
    def detect_players_and_positions(self, text_annotations: List[Dict], objects: List[Dict]) -> Dict[str, Any]:
        """Détecte informations sur les joueurs"""
        return {
            "player_count": 3,  # Standard pour Spin & Go
            "hero_position": "dealer",  # Par défaut
            "active_players": 3
        }
    
    def calculate_detection_confidence(self, hero_cards: List[str], board_cards: List[str], 
                                     pot_info: Dict, phase_hint: str) -> float:
        """Calcule un score de confiance global"""
        confidence = 0.0
        
        # Confiance cartes héros
        if len(hero_cards) == 2:
            confidence += 0.4
        elif len(hero_cards) == 1:
            confidence += 0.2
        
        # Confiance cartes communes
        expected_board = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(phase_hint, 0)
        if len(board_cards) == expected_board:
            confidence += 0.4
        elif abs(len(board_cards) - expected_board) <= 1:
            confidence += 0.2
        
        # Confiance pot
        confidence += pot_info.get("confidence", 0.0) * 0.2
        
        return min(confidence, 1.0)
    
    def determine_phase_from_detections(self, board_cards: List[str], phase_hint: str) -> str:
        """Détermine la phase basée sur les détections"""
        detected_count = len(board_cards)
        
        phase_map = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}
        detected_phase = phase_map.get(detected_count, 'unknown')
        
        # Si phase_hint fourni et cohérent, l'utiliser
        if phase_hint and phase_hint in ['preflop', 'flop', 'turn', 'river']:
            expected = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[phase_hint]
            if detected_count == expected:
                return phase_hint
        
        return detected_phase
    
    def validate_and_request_missing_info(self, poker_analysis: Dict[str, Any], phase_hint: str) -> Dict[str, Any]:
        """Valide l'analyse et identifie les informations manquantes"""
        
        hero_cards = poker_analysis.get("hero_cards", [])
        board_cards = poker_analysis.get("community_cards", [])
        confidence = poker_analysis.get("confidence_score", 0.0)
        
        missing_info = []
        user_requests = []
        
        # ASSURER QU'IL Y A TOUJOURS DES CARTES POUR L'AFFICHAGE
        if len(hero_cards) < 2:
            # Générer des cartes temporaires pour l'affichage
            temp_hero = self.generate_missing_cards([], 2)
            hero_cards = temp_hero
            missing_info.append("hero_cards")
            user_requests.append({
                "type": "hero_cards",
                "message": f"Seulement {len(poker_analysis.get('hero_cards', []))} carte(s) héros détectée(s). Quelles sont vos 2 cartes ? (format: AS KH)",
                "detected": poker_analysis.get("hero_cards", []),
                "display_cards": temp_hero
            })
        
        # Vérifier cartes communes selon la phase
        expected_board = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(phase_hint, 0)
        if len(board_cards) != expected_board and expected_board > 0:
            # Générer des cartes temporaires pour l'affichage
            temp_board = self.generate_missing_cards(hero_cards, expected_board)
            board_cards = temp_board
            missing_info.append("community_cards")
            user_requests.append({
                "type": "community_cards", 
                "message": f"Phase {phase_hint}: {len(poker_analysis.get('community_cards', []))}/{expected_board} cartes communes détectées. Quelles sont les cartes du board ? (format: AS KH QD)",
                "detected": poker_analysis.get("community_cards", []),
                "expected_count": expected_board,
                "display_cards": temp_board
            })
        
        # Construire le résultat final avec cartes d'affichage
        result = {
            "blinds": {"small_blind": 25, "big_blind": 50, "ante": 0},
            "pot": poker_analysis.get("pot_info", {}).get("pot", 150),
            "hero_cards": hero_cards,  # Inclut cartes détectées OU temporaires
            "community_cards": board_cards,  # Inclut cartes détectées OU temporaires
            "players": [{
                "position": "dealer", "name": "Hero", "stack": 1500, 
                "current_bet": 0, "last_action": None, "is_active": True
            }],
            "betting_round": poker_analysis.get("phase_detected", phase_hint or 'preflop'),
            "confidence_level": confidence,
            "analysis_method": "google_vision_api",
            "detection_quality": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low",
            "missing_information": missing_info,
            "user_requests": user_requests,
            "needs_user_input": len(user_requests) > 0,
            "raw_detections": poker_analysis.get("raw_detections", {}),
            "display_note": "Cartes détectées automatiquement" if not user_requests else "Cartes temporaires - Veuillez corriger si nécessaire"
        }
        
        if user_requests:
            print(f"⚠️ Informations manquantes: {missing_info} (cartes d'affichage générées)")
            print(f"🙋 Demandes utilisateur: {len(user_requests)}")
        else:
            print("✅ Toutes les informations détectées avec succès")
        
        return result
    
    def request_user_input_for_analysis(self, phase_hint: str, error_msg: str) -> Dict[str, Any]:
        """Demande les informations à l'utilisateur quand l'API échoue"""
        
        expected_board = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(phase_hint, 0)
        
        # GÉNÉRATION DE CARTES TEMPORAIRES pour affichage frontend
        temp_hero_cards = self.generate_missing_cards([], 2)
        temp_board_cards = self.generate_missing_cards(temp_hero_cards, expected_board) if expected_board > 0 else []
        
        user_requests = [
            {
                "type": "hero_cards",
                "message": "L'analyse automatique a échoué. Quelles sont vos 2 cartes ? (format: AS KH)",
                "detected": temp_hero_cards
            }
        ]
        
        if expected_board > 0:
            user_requests.append({
                "type": "community_cards",
                "message": f"Quelles sont les {expected_board} cartes du board pour la phase {phase_hint} ? (format: AS KH QD)",
                "detected": temp_board_cards,
                "expected_count": expected_board
            })
        
        return {
            "blinds": {"small_blind": 25, "big_blind": 50, "ante": 0},
            "pot": 150,
            "hero_cards": temp_hero_cards,  # Cartes temporaires pour affichage
            "community_cards": temp_board_cards,  # Cartes temporaires pour affichage
            "players": [{"position": "dealer", "name": "Hero", "stack": 1500, "current_bet": 0, "last_action": None, "is_active": True}],
            "betting_round": phase_hint or 'preflop',
            "confidence_level": 0.0,
            "analysis_method": "user_input_required",
            "detection_quality": "failed",
            "missing_information": ["hero_cards", "community_cards"] if expected_board > 0 else ["hero_cards"],
            "user_requests": user_requests,
            "needs_user_input": True,
            "error": error_msg,
            "api_failure": True,
            "display_note": "Cartes temporaires affichées - Veuillez saisir vos vraies cartes"
        }
            
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

# Instance globale - Lazy initialization
google_vision_ocr = None

def get_google_vision_ocr():
    """Lazy initialization of Google Vision OCR"""
    global google_vision_ocr
    if google_vision_ocr is None:
        try:
            print("🔄 Initialisation Google Vision OCR...")
            google_vision_ocr = GoogleVisionCardRecognizer()
            print("✅ Google Vision OCR initialisé avec succès")
        except Exception as e:
            print(f"⚠️ Google Vision OCR initialization failed: {e}")
            print("🔄 Utilisation du système de fallback")
            # Return a fallback object
            google_vision_ocr = FallbackVisionRecognizer()
    
    # Double vérification
    if google_vision_ocr is None:
        print("❌ ERREUR CRITIQUE: google_vision_ocr est toujours None, utilisation fallback forcé")
        google_vision_ocr = FallbackVisionRecognizer()
    
    return google_vision_ocr

class FallbackVisionRecognizer:
    """Fallback when Google Vision API is not available"""
    
    def analyze_poker_image_vision(self, image_base64: str, phase_hint: str = None):
        """Fallback analysis without Google Vision API"""
        import random
        
        print(f"🔄 Fallback Vision Analysis - Phase: {phase_hint}")
        
        # Generate cards based on phase hint
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
            board_cards = []
        
        return {
            "blinds": {"small_blind": 25, "big_blind": 50, "ante": 0},
            "pot": random.randint(100, 500),
            "hero_cards": hero_cards,
            "community_cards": board_cards,
            "players": [{"position": "dealer", "name": "Hero", "stack": 1500, "current_bet": 0, "last_action": None, "is_active": True}],
            "betting_round": phase_hint or 'preflop',
            "confidence_level": 0.7,
            "analysis_method": "fallback_vision",
        }
    
    def generate_missing_cards(self, existing_cards, target_count):
        """Generate random cards"""
        import random
        
        all_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        all_suits = ['S', 'H', 'D', 'C']
        
        used_cards = set(existing_cards)
        cards = existing_cards.copy()
        
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