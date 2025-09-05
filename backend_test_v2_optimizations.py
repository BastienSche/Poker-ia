#!/usr/bin/env python3
"""
Tests spécialisés pour les optimisations v2.0 du Poker Assistant Pro
Focus sur la performance, précision et cache
"""
import requests
import json
import base64
import sys
import time
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

class PokerAssistantV2Tester:
    def __init__(self, base_url="https://pokeranalyzer-1.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_id = f"v2_test_session_{int(time.time())}"
        self.tests_run = 0
        self.tests_passed = 0
        self.errors = []
        self.performance_data = []

    def log_test(self, name, success, details=""):
        """Enregistre le résultat d'un test"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - RÉUSSI")
        else:
            print(f"❌ {name} - ÉCHEC: {details}")
            self.errors.append(f"{name}: {details}")
        
        if details and success:
            print(f"   ℹ️  {details}")

    def test_v2_api_version(self):
        """Test que l'API retourne bien la version 2.0.0 et optimized: true"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                version = data.get("version")
                optimized = data.get("optimized")
                
                if version == "2.0.0" and optimized is True:
                    self.log_test("Version API v2.0", True, 
                                f"Version: {version}, Optimisé: {optimized}")
                    return True
                else:
                    self.log_test("Version API v2.0", False, 
                                f"Version: {version}, Optimisé: {optimized}")
                    return False
            else:
                self.log_test("Version API v2.0", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Version API v2.0", False, f"Exception: {str(e)}")
            return False

    def create_poker_test_image(self, with_cards=True):
        """Crée une image de test simulant une table de poker avec cartes"""
        img = Image.new('RGB', (1280, 720), color=(0, 100, 0))  # Fond vert poker
        draw = ImageDraw.Draw(img)
        
        if with_cards:
            # Simulation de cartes de poker
            # Cartes du joueur (hero cards)
            draw.rectangle([100, 500, 180, 600], fill='white', outline='black', width=2)
            draw.text((110, 520), "AS", fill='black')  # As de Pique
            
            draw.rectangle([200, 500, 280, 600], fill='white', outline='black', width=2)
            draw.text((210, 520), "KH", fill='red')   # Roi de Coeur
            
            # Cartes communes (community cards)
            for i, card in enumerate(["QD", "JC", "TS"]):
                x = 400 + i * 100
                draw.rectangle([x, 300, x+80, 400], fill='white', outline='black', width=2)
                color = 'red' if card[1] in ['D', 'H'] else 'black'
                draw.text((x+10, 320), card, fill=color)
            
            # Pot
            draw.ellipse([550, 200, 650, 250], fill='yellow', outline='black', width=2)
            draw.text((570, 215), "POT", fill='black')
            
            # Blinds
            draw.text((50, 50), "SB: 25", fill='white')
            draw.text((50, 80), "BB: 50", fill='white')
        
        # Conversion en base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=75)  # Compression 75% comme dans le frontend
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

    def test_performance_analysis(self):
        """Test de performance - l'analyse doit être < 5s"""
        try:
            test_image_b64 = self.create_poker_test_image()
            
            payload = {
                "image_base64": test_image_b64,
                "session_id": self.session_id
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            self.performance_data.append(processing_time)
            
            if response.status_code == 200:
                data = response.json()
                api_processing_time = data.get('processing_time', 0)
                
                if processing_time < 5.0:
                    self.log_test("Performance < 5s", True, 
                                f"Temps total: {processing_time:.2f}s, API: {api_processing_time:.2f}s")
                    return True
                else:
                    self.log_test("Performance < 5s", False, 
                                f"Temps trop long: {processing_time:.2f}s")
                    return False
            else:
                self.log_test("Performance < 5s", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Performance < 5s", False, f"Exception: {str(e)}")
            return False

    def test_cache_functionality(self):
        """Test du cache - deuxième appel identique doit être < 0.1s"""
        try:
            test_image_b64 = self.create_poker_test_image()
            
            payload = {
                "image_base64": test_image_b64,
                "session_id": self.session_id
            }
            
            # Premier appel
            start_time = time.time()
            response1 = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            first_call_time = time.time() - start_time
            
            if response1.status_code != 200:
                self.log_test("Cache - Premier appel", False, f"Status {response1.status_code}")
                return False
            
            # Deuxième appel identique (devrait utiliser le cache)
            start_time = time.time()
            response2 = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            second_call_time = time.time() - start_time
            
            if response2.status_code == 200:
                data2 = response2.json()
                api_processing_time = data2.get('processing_time', 0)
                
                # Le cache devrait rendre le deuxième appel beaucoup plus rapide
                if api_processing_time <= 0.1:
                    self.log_test("Cache fonctionnel", True, 
                                f"1er appel: {first_call_time:.2f}s, 2ème (cache): {api_processing_time:.3f}s")
                    return True
                else:
                    self.log_test("Cache fonctionnel", False, 
                                f"Cache pas assez rapide: {api_processing_time:.3f}s")
                    return False
            else:
                self.log_test("Cache fonctionnel", False, f"Status {response2.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Cache fonctionnel", False, f"Exception: {str(e)}")
            return False

    def test_card_format_validation(self):
        """Test de la validation du format des cartes (AS, KH, QD, etc.)"""
        try:
            test_image_b64 = self.create_poker_test_image()
            
            payload = {
                "image_base64": test_image_b64,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_elements = data.get('detected_elements', {})
                
                # Vérification des cartes du héros
                hero_cards = detected_elements.get('hero_cards', [])
                community_cards = detected_elements.get('community_cards', [])
                
                valid_format = True
                invalid_cards = []
                
                # Validation du format des cartes (2 caractères: rang + couleur)
                all_cards = hero_cards + community_cards
                for card in all_cards:
                    if not isinstance(card, str) or len(card) != 2:
                        valid_format = False
                        invalid_cards.append(card)
                        continue
                    
                    rank = card[0].upper()
                    suit = card[1].upper()
                    
                    valid_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
                    valid_suits = ['S', 'H', 'D', 'C']
                    
                    if rank not in valid_ranks or suit not in valid_suits:
                        valid_format = False
                        invalid_cards.append(card)
                
                if valid_format and len(all_cards) > 0:
                    self.log_test("Format des cartes", True, 
                                f"Cartes détectées: {all_cards}")
                    return True
                elif len(all_cards) == 0:
                    self.log_test("Format des cartes", True, 
                                "Aucune carte détectée (normal pour image de test)")
                    return True
                else:
                    self.log_test("Format des cartes", False, 
                                f"Cartes invalides: {invalid_cards}")
                    return False
            else:
                self.log_test("Format des cartes", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Format des cartes", False, f"Exception: {str(e)}")
            return False

    def test_json_parsing_robustness(self):
        """Test de la robustesse du parsing JSON"""
        try:
            # Test avec une image vide pour forcer une réponse d'erreur contrôlée
            empty_image = base64.b64encode(b"invalid_image_data").decode('utf-8')
            
            payload = {
                "image_base64": empty_image,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # L'API devrait retourner une structure valide même en cas d'erreur
                required_fields = ['session_id', 'detected_elements']
                has_required_fields = all(field in data for field in required_fields)
                
                if has_required_fields:
                    self.log_test("Robustesse JSON", True, 
                                "Structure JSON valide même avec image invalide")
                    return True
                else:
                    self.log_test("Robustesse JSON", False, 
                                f"Champs manquants dans la réponse: {data}")
                    return False
            else:
                # Un code d'erreur est acceptable pour une image invalide
                self.log_test("Robustesse JSON", True, 
                            f"Gestion d'erreur appropriée: {response.status_code}")
                return True
                
        except Exception as e:
            self.log_test("Robustesse JSON", False, f"Exception: {str(e)}")
            return False

    def test_performance_stats_endpoint(self):
        """Test de l'endpoint des statistiques de performance"""
        try:
            response = requests.get(f"{self.api_url}/performance/stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['cache_size', 'average_processing_time', 'optimization_level', 'version']
                
                missing_fields = [field for field in expected_fields if field not in data]
                if missing_fields:
                    self.log_test("Stats de performance", False, 
                                f"Champs manquants: {missing_fields}")
                    return False
                
                if data.get('version') == '2.0.0' and data.get('optimization_level') == 'high':
                    self.log_test("Stats de performance", True, 
                                f"Cache: {data['cache_size']}, Temps moy: {data['average_processing_time']}")
                    return True
                else:
                    self.log_test("Stats de performance", False, 
                                f"Version ou niveau d'optimisation incorrect: {data}")
                    return False
            else:
                self.log_test("Stats de performance", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Stats de performance", False, f"Exception: {str(e)}")
            return False

    def run_v2_optimization_tests(self):
        """Exécute tous les tests d'optimisation v2.0"""
        print("🚀 Tests des Optimisations v2.0 - Poker Assistant Pro")
        print(f"📍 URL de base: {self.base_url}")
        print(f"🆔 Session de test: {self.session_id}")
        print("=" * 70)
        
        # Tests spécifiques v2.0
        self.test_v2_api_version()
        self.test_performance_analysis()
        self.test_cache_functionality()
        self.test_card_format_validation()
        self.test_json_parsing_robustness()
        self.test_performance_stats_endpoint()
        
        # Résumé des performances
        if self.performance_data:
            avg_time = sum(self.performance_data) / len(self.performance_data)
            print(f"\n⚡ PERFORMANCE MOYENNE: {avg_time:.2f}s")
        
        # Résumé final
        print("\n" + "=" * 70)
        print(f"📊 RÉSULTATS v2.0: {self.tests_passed}/{self.tests_run} tests réussis")
        
        if self.errors:
            print("\n❌ PROBLÈMES DÉTECTÉS:")
            for error in self.errors:
                print(f"   • {error}")
        else:
            print("\n✅ Toutes les optimisations v2.0 fonctionnent parfaitement!")
        
        return len(self.errors) == 0

def main():
    """Point d'entrée principal"""
    tester = PokerAssistantV2Tester()
    success = tester.run_v2_optimization_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())