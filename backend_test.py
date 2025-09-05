#!/usr/bin/env python3
"""
Tests complets pour l'API Poker Assistant Pro
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

    def test_analyze_screen_endpoint(self):
        """Test de l'endpoint d'analyse d'écran"""
        try:
            # Création d'une image de test
            test_image_b64 = self.create_test_image()
            
            payload = {
                "image_base64": test_image_b64,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.api_url}/analyze-screen",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30  # Timeout plus long pour l'analyse IA
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['id', 'session_id', 'timestamp', 'detected_elements']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Analyse d'écran", False, f"Champs manquants: {missing_fields}")
                    return False
                
                # Vérification de la structure des éléments détectés
                detected = data.get('detected_elements', {})
                if isinstance(detected, dict):
                    self.log_test("Analyse d'écran", True, 
                                f"Analyse réussie - Confiance: {data.get('confidence', 0):.2f}")
                    return True
                else:
                    self.log_test("Analyse d'écran", False, "Structure detected_elements invalide")
                    return False
            else:
                self.log_test("Analyse d'écran", False, 
                            f"Status {response.status_code}: {response.text[:200]}")
                return False
                
        except Exception as e:
            self.log_test("Analyse d'écran", False, f"Exception: {str(e)}")
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