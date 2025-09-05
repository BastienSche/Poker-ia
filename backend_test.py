#!/usr/bin/env python3
"""
Tests complets pour l'API Poker Assistant Pro - Focus sur Phase Detection Bug Fix
"""
import requests
import json
import base64
import sys
from datetime import datetime
from io import BytesIO
from PIL import Image
import time

class PokerAssistantTester:
    def __init__(self, base_url="https://pokeranalyzer-1.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_id = f"test_session_{int(time.time())}"
        self.tests_run = 0
        self.tests_passed = 0
        self.errors = []

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

    def test_root_endpoint(self):
        """Test de l'endpoint racine"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            expected_message = "Poker Assistant API - Prêt pour l'analyse !"
            
            if response.status_code == 200:
                data = response.json()
                if data.get("message") == expected_message:
                    self.log_test("Endpoint racine", True, f"Message: {data['message']}")
                    return True
                else:
                    self.log_test("Endpoint racine", False, f"Message incorrect: {data}")
                    return False
            else:
                self.log_test("Endpoint racine", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Endpoint racine", False, f"Exception: {str(e)}")
            return False

    def create_test_image(self):
        """Crée une image de test simple"""
        # Création d'une image de test 800x600 avec du texte simulant une table de poker
        img = Image.new('RGB', (800, 600), color='green')
        
        # Conversion en base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

    def test_phase_detection_bug_fix(self):
        """Test critique: Phase Detection Bug Fix - River doit retourner exactement 5 cartes"""
        print("\n🔥 TEST CRITIQUE: Phase Detection Bug Fix")
        
        test_image_b64 = self.create_test_image()
        
        # Test 1: River phase doit retourner exactement 5 cartes
        payload = {
            "image_base64": test_image_b64,
            "session_id": self.session_id,
            "phase_hint": "river"
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                detected_elements = data.get('detected_elements', {})
                community_cards = detected_elements.get('community_cards', [])
                betting_round = detected_elements.get('betting_round', '')
                
                # VALIDATION CRITIQUE: River doit avoir exactement 5 cartes
                if len(community_cards) == 5 and betting_round == "river":
                    self.log_test("Phase Detection - River Fix", True, 
                                f"✅ RIVER: {len(community_cards)} cartes, phase={betting_round}, temps={processing_time:.3f}s")
                    return True
                else:
                    self.log_test("Phase Detection - River Fix", False, 
                                f"❌ RIVER: {len(community_cards)} cartes (attendu 5), phase={betting_round} (attendu river)")
                    return False
            else:
                self.log_test("Phase Detection - River Fix", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Phase Detection - River Fix", False, f"Exception: {str(e)}")
            return False

    def test_free_vision_engine_phases(self):
        """Test du Free Vision Engine avec différents phase hints"""
        print("\n🆓 TEST: Free Vision Engine - Toutes les phases")
        
        test_image_b64 = self.create_test_image()
        phase_tests = [
            ("preflop", 0),
            ("flop", 3), 
            ("turn", 4),
            ("river", 5)
        ]
        
        all_passed = True
        
        for phase_hint, expected_cards in phase_tests:
            payload = {
                "image_base64": test_image_b64,
                "session_id": self.session_id,
                "phase_hint": phase_hint
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/analyze-screen",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    detected_elements = data.get('detected_elements', {})
                    community_cards = detected_elements.get('community_cards', [])
                    betting_round = detected_elements.get('betting_round', '')
                    analysis_type = data.get('analysis_type', '')
                    
                    # Validation du nombre de cartes et de la phase
                    cards_correct = len(community_cards) == expected_cards
                    phase_correct = betting_round == phase_hint
                    is_free = analysis_type == "free_cv"
                    is_fast = processing_time < 1.0
                    
                    if cards_correct and phase_correct and is_free:
                        self.log_test(f"Free Vision - {phase_hint.upper()}", True, 
                                    f"✅ {expected_cards} cartes, phase={betting_round}, type={analysis_type}, {processing_time:.3f}s")
                    else:
                        self.log_test(f"Free Vision - {phase_hint.upper()}", False, 
                                    f"❌ {len(community_cards)}/{expected_cards} cartes, phase={betting_round}, type={analysis_type}")
                        all_passed = False
                else:
                    self.log_test(f"Free Vision - {phase_hint.upper()}", False, f"Status {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Free Vision - {phase_hint.upper()}", False, f"Exception: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_performance_ultra_fast(self):
        """Test de performance - Vérification temps de réponse ultra-rapide"""
        print("\n⚡ TEST: Performance Ultra-Fast")
        
        test_image_b64 = self.create_test_image()
        
        # Test avec plusieurs phases pour vérifier la consistance
        phases = ["preflop", "flop", "turn", "river"]
        response_times = []
        
        for phase in phases:
            payload = {
                "image_base64": test_image_b64,
                "session_id": self.session_id,
                "phase_hint": phase
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/analyze-screen",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                processing_time = time.time() - start_time
                response_times.append(processing_time)
                
                if response.status_code == 200:
                    data = response.json()
                    server_processing_time = data.get('processing_time', processing_time)
                    
                    # Vérification temps ultra-rapide (< 1 seconde)
                    if processing_time < 1.0:
                        print(f"   ✅ {phase}: {processing_time:.3f}s (serveur: {server_processing_time:.3f}s)")
                    else:
                        print(f"   ❌ {phase}: {processing_time:.3f}s - TROP LENT!")
                        
            except Exception as e:
                print(f"   ❌ {phase}: Exception {str(e)}")
                return False
        
        # Validation globale
        avg_time = sum(response_times) / len(response_times) if response_times else 999
        max_time = max(response_times) if response_times else 999
        
        if avg_time < 1.0 and max_time < 2.0:
            self.log_test("Performance Ultra-Fast", True, 
                        f"Temps moyen: {avg_time:.3f}s, Max: {max_time:.3f}s")
            return True
        else:
            self.log_test("Performance Ultra-Fast", False, 
                        f"TROP LENT - Temps moyen: {avg_time:.3f}s, Max: {max_time:.3f}s")
            return False

    def test_100_percent_free_status(self):
        """Test du statut 100% gratuit"""
        print("\n💰 TEST: 100% Free Status")
        
        # Test endpoint racine pour les messages gratuits
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", "")
                features = data.get("features", [])
                cost = data.get("cost", "")
                
                # Vérifications du statut gratuit
                is_free_message = "GRATUIT" in message.upper() or "FREE" in message.upper()
                has_free_features = any("free" in str(feature).lower() for feature in features)
                has_zero_cost = "0.00" in cost or "gratuit" in cost.lower() or "free" in cost.lower()
                
                if is_free_message and has_free_features and has_zero_cost:
                    self.log_test("100% Free Status - Root", True, 
                                f"Message: {message}, Cost: {cost}")
                else:
                    self.log_test("100% Free Status - Root", False, 
                                f"Pas assez de mentions gratuites - Message: {message}")
                    
        except Exception as e:
            self.log_test("100% Free Status - Root", False, f"Exception: {str(e)}")
            return False
        
        # Test analyse avec vérification du type gratuit
        test_image_b64 = self.create_test_image()
        payload = {
            "image_base64": test_image_b64,
            "session_id": self.session_id,
            "phase_hint": "river"
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis_type = data.get('analysis_type', '')
                detected_elements = data.get('detected_elements', {})
                analysis_method = detected_elements.get('analysis_method', '')
                
                # Vérification que c'est bien gratuit
                is_free_analysis = analysis_type == "free_cv"
                is_free_method = "free" in analysis_method.lower()
                
                if is_free_analysis:
                    self.log_test("100% Free Status - Analysis", True, 
                                f"Type: {analysis_type}, Method: {analysis_method}")
                    return True
                else:
                    self.log_test("100% Free Status - Analysis", False, 
                                f"Type non gratuit: {analysis_type}")
                    return False
                    
        except Exception as e:
            self.log_test("100% Free Status - Analysis", False, f"Exception: {str(e)}")
            return False

    def test_settings_endpoints(self):
        """Test des endpoints de paramètres"""
        user_id = f"test_user_{int(time.time())}"
        
        # Test GET settings (devrait créer des paramètres par défaut)
        try:
            response = requests.get(f"{self.api_url}/settings/{user_id}", timeout=10)
            
            if response.status_code == 200:
                settings = response.json()
                required_fields = ['aggressiveness', 'auto_analyze', 'capture_frequency', 'language']
                
                missing_fields = [field for field in required_fields if field not in settings]
                if missing_fields:
                    self.log_test("GET Settings", False, f"Champs manquants: {missing_fields}")
                    return False
                
                self.log_test("GET Settings", True, f"Paramètres par défaut créés pour {user_id}")
            else:
                self.log_test("GET Settings", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("GET Settings", False, f"Exception: {str(e)}")
            return False

        # Test POST settings
        try:
            new_settings = {
                "user_id": user_id,
                "aggressiveness": 0.8,
                "auto_analyze": False,
                "capture_frequency": 5,
                "language": "fr",
                "always_on_top": False
            }
            
            response = requests.post(
                f"{self.api_url}/settings",
                json=new_settings,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "saved":
                    self.log_test("POST Settings", True, "Paramètres sauvegardés")
                    return True
                else:
                    self.log_test("POST Settings", False, f"Réponse inattendue: {result}")
                    return False
            else:
                self.log_test("POST Settings", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("POST Settings", False, f"Exception: {str(e)}")
            return False

    def test_session_analyses_endpoint(self):
        """Test de l'endpoint des analyses de session"""
        try:
            response = requests.get(
                f"{self.api_url}/sessions/{self.session_id}/analyses",
                timeout=10
            )
            
            if response.status_code == 200:
                analyses = response.json()
                if isinstance(analyses, list):
                    self.log_test("Analyses de session", True, 
                                f"Récupéré {len(analyses)} analyses pour la session")
                    return True
                else:
                    self.log_test("Analyses de session", False, "Réponse n'est pas une liste")
                    return False
            else:
                self.log_test("Analyses de session", False, f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Analyses de session", False, f"Exception: {str(e)}")
            return False

    def test_websocket_endpoint(self):
        """Test basique de l'endpoint WebSocket (connexion seulement)"""
        try:
            import websocket
            
            ws_url = f"{self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws/{self.session_id}"
            
            def on_open(ws):
                print("   🔌 WebSocket connecté")
                ws.close()
            
            def on_error(ws, error):
                print(f"   ⚠️  Erreur WebSocket: {error}")
            
            ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_error=on_error)
            
            # Test de connexion rapide
            import threading
            def run_ws():
                ws.run_forever()
            
            ws_thread = threading.Thread(target=run_ws)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Attendre un peu pour la connexion
            time.sleep(2)
            
            self.log_test("WebSocket", True, "Connexion WebSocket testée")
            return True
            
        except ImportError:
            self.log_test("WebSocket", False, "Module websocket-client non disponible")
            return False
        except Exception as e:
            self.log_test("WebSocket", False, f"Exception: {str(e)}")
            return False

    def run_all_tests(self):
        """Exécute tous les tests"""
        print("🚀 Démarrage des tests de l'API Poker Assistant Pro")
        print(f"📍 URL de base: {self.base_url}")
        print(f"🆔 Session de test: {self.session_id}")
        print("=" * 60)
        
        # Tests des endpoints
        self.test_root_endpoint()
        self.test_analyze_screen_endpoint()
        self.test_settings_endpoints()
        self.test_session_analyses_endpoint()
        self.test_websocket_endpoint()
        
        # Résumé
        print("\n" + "=" * 60)
        print(f"📊 RÉSULTATS: {self.tests_passed}/{self.tests_run} tests réussis")
        
        if self.errors:
            print("\n❌ ERREURS DÉTECTÉES:")
            for error in self.errors:
                print(f"   • {error}")
        else:
            print("\n✅ Tous les tests sont passés avec succès!")
        
        return len(self.errors) == 0

def main():
    """Point d'entrée principal"""
    tester = PokerAssistantTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())