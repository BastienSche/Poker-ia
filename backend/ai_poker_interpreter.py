"""
Interpréteur IA pour analyser les résultats de Google Vision API
Utilise une IA générative pour transformer le texte brut en données structurées de poker
"""
import os
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from emergentintegrations.llm.chat import LlmChat, UserMessage

# Charger les variables d'environnement
load_dotenv()

class AIPokerInterpreter:
    """Interprète intelligemment les résultats OCR pour le poker"""
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not self.api_key:
            raise ValueError("EMERGENT_LLM_KEY non trouvée dans les variables d'environnement")
        
        print(f"🤖 AI Interpreter initialisé avec clé: ***{self.api_key[-4:]}")
        
        # Système prompt spécialisé pour le poker
        self.system_prompt = """Tu es un expert en analyse de données de poker OCR. 

TON RÔLE:
- Analyser le texte brut détecté par Google Vision API sur des tables de poker
- Identifier et corriger les erreurs d'OCR courantes
- Extraire les cartes de poker du texte désordonné
- Distinguer les cartes du héros vs cartes communes (board)
- Retourner des données JSON parfaitement structurées

CARTES DE POKER:
- Rangs: A, K, Q, J, T (10), 9, 8, 7, 6, 5, 4, 3, 2
- Couleurs: S (♠), H (♥), D (♦), C (♣)
- Format: "AS" = As de Pique, "KH" = Roi de Cœur

ERREURS OCR COURANTES À CORRIGER:
- "◆" → "D" (Diamant)
- "♦" → "D" (Diamant)  
- "♠" → "S" (Pique)
- "♥" → "H" (Cœur)
- "♣" → "C" (Trèfle)
- "10" → "T" (Dix)
- "0" → "O" puis "Q" (confusion OCR)
- "1" → "I" puis "J" (confusion OCR)

RÉPONSE REQUISE:
Retourne SEULEMENT un JSON valide dans ce format exact:
{
  "hero_cards": ["AS", "KH"],
  "community_cards": ["QD", "JC", "TS"],
  "confidence": 0.9,
  "interpretation_notes": "Texte explicatif de ce qui a été interprété",
  "corrections_made": ["◆ corrigé en D", "♥ corrigé en H"]
}

IMPORTANT: Si tu ne peux pas identifier de cartes, retourne des listes vides [] mais garde la structure JSON."""
    
    async def interpret_ocr_results(self, ocr_text: str, phase_hint: str, position_info: List[Dict] = None) -> Dict[str, Any]:
        """
        Interprète les résultats OCR avec l'IA générative
        
        Args:
            ocr_text: Texte brut de Google Vision API
            phase_hint: Phase du jeu (preflop, flop, turn, river)
            position_info: Informations de position des textes détectés
            
        Returns:
            Dict avec cartes structurées et métadonnées
        """
        try:
            print(f"🤖 === INTERPRÉTATION IA === Phase: {phase_hint}")
            print(f"📝 Texte à analyser: '{ocr_text[:200]}...'")
            
            # Créer le chat LLM
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"poker_ocr_{phase_hint}",
                system_message=self.system_prompt
            ).with_model("openai", "gpt-4o-mini")  # Modèle rapide et efficace
            
            # Construire le prompt d'analyse
            analysis_prompt = self.build_analysis_prompt(ocr_text, phase_hint, position_info)
            
            # Envoyer à l'IA
            user_message = UserMessage(text=analysis_prompt)
            
            print("🚀 Envoi à l'IA générative...")
            response = await chat.send_message(user_message)
            
            print(f"🤖 Réponse IA reçue: {len(response)} caractères")
            
            # Parser la réponse JSON
            parsed_result = self.parse_ai_response(response)
            
            print(f"✅ IA INTERPRÉTATION: Hero={len(parsed_result.get('hero_cards', []))}, Board={len(parsed_result.get('community_cards', []))}")
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Erreur interprétation IA: {e}")
            return self.get_fallback_interpretation(ocr_text, phase_hint)
    
    def build_analysis_prompt(self, ocr_text: str, phase_hint: str, position_info: List[Dict] = None) -> str:
        """Construit le prompt d'analyse pour l'IA"""
        
        expected_cards = {
            'preflop': {'hero': 2, 'board': 0},
            'flop': {'hero': 2, 'board': 3},
            'turn': {'hero': 2, 'board': 4},
            'river': {'hero': 2, 'board': 5}
        }
        
        expected = expected_cards.get(phase_hint, {'hero': 2, 'board': 0})
        
        prompt = f"""ANALYSE CE TEXTE OCR D'UNE TABLE DE POKER:

TEXTE BRUT DÉTECTÉ:
"{ocr_text}"

CONTEXTE:
- Phase de jeu: {phase_hint}
- Cartes héros attendues: {expected['hero']}
- Cartes board attendues: {expected['board']}

INFORMATIONS SUPPLÉMENTAIRES:"""
        
        if position_info:
            prompt += f"""
POSITIONS DES TEXTES DÉTECTÉS:
"""
            for i, pos in enumerate(position_info[:10]):  # Limiter à 10
                text = pos.get('text', '')
                x = pos.get('x', 0)
                y = pos.get('y', 0)
                prompt += f"  - '{text}' à position ({x:.0f}, {y:.0f})\n"
        
        prompt += f"""

TÂCHE:
1. Identifie toutes les cartes de poker dans ce texte
2. Corrige les erreurs d'OCR courantes (◆→D, ♥→H, etc.)
3. Sépare les cartes héros (généralement en bas) des cartes communes (généralement au centre)
4. Pour {phase_hint}: attends {expected['hero']} cartes héros et {expected['board']} cartes board

RETOURNE UNIQUEMENT LE JSON STRUCTURÉ (pas d'explication supplémentaire)."""
        
        return prompt
    
    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse la réponse JSON de l'IA"""
        try:
            import json
            
            # Nettoyer la réponse (enlever markdown, etc.)
            cleaned = response.strip()
            
            # Si la réponse contient ```json, l'extraire
            if "```json" in cleaned:
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                if end != -1:
                    cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                start = cleaned.find("```") + 3
                end = cleaned.find("```", start)
                if end != -1:
                    cleaned = cleaned[start:end].strip()
            
            # Parser le JSON
            parsed = json.loads(cleaned)
            
            # Valider la structure
            required_keys = ['hero_cards', 'community_cards', 'confidence']
            for key in required_keys:
                if key not in parsed:
                    parsed[key] = [] if 'cards' in key else 0.0
            
            # Valider les cartes
            parsed['hero_cards'] = self.validate_cards(parsed.get('hero_cards', []))
            parsed['community_cards'] = self.validate_cards(parsed.get('community_cards', []))
            
            return parsed
            
        except Exception as e:
            print(f"❌ Erreur parsing réponse IA: {e}")
            print(f"Réponse brute: {response[:200]}...")
            
            return {
                "hero_cards": [],
                "community_cards": [],
                "confidence": 0.0,
                "interpretation_notes": f"Erreur parsing: {str(e)}",
                "corrections_made": [],
                "parse_error": True
            }
    
    def validate_cards(self, cards: List[str]) -> List[str]:
        """Valide et nettoie une liste de cartes"""
        valid_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        valid_suits = ['S', 'H', 'D', 'C']
        
        validated = []
        
        for card in cards:
            if not isinstance(card, str) or len(card) != 2:
                continue
                
            rank, suit = card[0].upper(), card[1].upper()
            
            if rank in valid_ranks and suit in valid_suits:
                validated.append(f"{rank}{suit}")
        
        return validated
    
    def get_fallback_interpretation(self, ocr_text: str, phase_hint: str) -> Dict[str, Any]:
        """Interprétation de fallback si l'IA échoue"""
        
        print("🔄 Fallback: Interprétation basique sans IA")
        
        # Extraction basique avec regex
        import re
        
        # Chercher patterns de cartes
        card_pattern = r'([AKQJT2-9]|10)([♠♥♦♣SHDC])'
        matches = re.findall(card_pattern, ocr_text.upper())
        
        basic_cards = []
        for rank, suit in matches:
            # Corrections basiques
            if suit == '♠': suit = 'S'
            elif suit == '♥': suit = 'H'
            elif suit == '♦': suit = 'D'
            elif suit == '♣': suit = 'C'
            elif rank == '10': rank = 'T'
            
            if len(rank) == 1 and len(suit) == 1:
                basic_cards.append(f"{rank}{suit}")
        
        # Répartition basique
        expected_board = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(phase_hint, 0)
        
        hero_cards = basic_cards[:2] if len(basic_cards) >= 2 else basic_cards
        board_cards = basic_cards[2:2+expected_board] if len(basic_cards) > 2 else []
        
        return {
            "hero_cards": hero_cards,
            "community_cards": board_cards,
            "confidence": 0.3,  # Confiance faible
            "interpretation_notes": "Fallback: Extraction regex basique sans IA",
            "corrections_made": ["Utilisé extraction basique"],
            "fallback_used": True
        }

# Instance globale
ai_interpreter = AIPokerInterpreter()

def get_ai_interpreter():
    """Retourne l'instance de l'interpréteur IA"""
    return ai_interpreter