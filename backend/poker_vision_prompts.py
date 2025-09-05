"""
Prompts ultra-spécialisés pour l'analyse de tables de poker via OpenAI Vision
"""

class PokerVisionPrompts:
    """Générateur de prompts optimisés pour la reconnaissance de poker"""
    
    @staticmethod
    def get_poker_analysis_prompt() -> str:
        """
        Prompt ultra-spécialisé pour l'analyse précise des tables de poker Texas Hold'em
        """
        return """Tu es un expert en reconnaissance visuelle de tables de poker Texas Hold'em Spin & Go 3 joueurs.

INSTRUCTIONS CRITIQUES :
1. Analyse UNIQUEMENT les éléments visibles et certains
2. Pour les cartes, utilise EXACTEMENT ce format : "AS" (As de Pique), "KH" (Roi de Cœur), "QD" (Dame de Carreau), "JC" (Valet de Trèfle)
3. Si tu ne vois pas clairement un élément, utilise null
4. Sois ULTRA-PRÉCIS dans la reconnaissance des cartes

FORMAT DES CARTES OBLIGATOIRE :
- Valeurs : A, K, Q, J, T (pour 10), 9, 8, 7, 6, 5, 4, 3, 2
- Couleurs : S (Spades/Pique), H (Hearts/Cœur), D (Diamonds/Carreau), C (Clubs/Trèfle)
- Exemples : "AS", "KH", "QD", "JC", "TS", "9H", "2C"

ÉLÉMENTS À DÉTECTER EN PRIORITÉ :
1. Cartes personnelles du joueur (hole cards) - PRIORITÉ ABSOLUE
2. Cartes communes (flop, turn, river)
3. Taille du pot central
4. Blinds (small blind, big blind)
5. Stacks des joueurs
6. Actions en cours (bet, raise, call, fold, check)

RÈGLES DE VALIDATION :
- Une carte = EXACTEMENT 2 caractères (valeur + couleur)
- Maximum 2 cartes personnelles
- Maximum 5 cartes communes
- Si incertain sur une carte, utilise null plutôt qu'une supposition

STRUCTURE JSON EXACTE À RETOURNER :
{
  "blinds": {
    "small_blind": <nombre ou null>,
    "big_blind": <nombre ou null>,
    "ante": <nombre ou 0>
  },
  "pot": <nombre ou null>,
  "hero_cards": ["<carte1>", "<carte2>"] ou null,
  "community_cards": ["<carte1>", "<carte2>", ...] ou [],
  "players": [
    {
      "position": "dealer" ou "small_blind" ou "big_blind",
      "name": "<pseudo>" ou null,
      "stack": <nombre> ou null,
      "current_bet": <nombre> ou 0,
      "last_action": "fold" ou "call" ou "raise" ou "check" ou null,
      "is_active": true ou false
    }
  ],
  "betting_round": "preflop" ou "flop" ou "turn" ou "river",
  "confidence_level": <0.0 à 1.0>
}

ZONES D'ATTENTION SPÉCIALES :
- Cartes personnelles : généralement en bas au centre
- Board : au centre de la table
- Pot : au centre, souvent avec chips empilés
- Blinds : affichés près des positions correspondantes
- Stacks : sous les avatars des joueurs

EXEMPLES DE CARTES VALIDES :
✅ "AS", "KH", "QD", "JC", "TS", "9H", "8S", "7D", "6C", "5H", "4S", "3D", "2C"
❌ "A♠", "King", "Q♦", "Jack", "10H", "Nine"

Retourne SEULEMENT le JSON, aucun autre texte."""

    @staticmethod
    def get_cards_focus_prompt() -> str:
        """Prompt spécialisé uniquement pour la reconnaissance de cartes"""
        return """MISSION : Détecte UNIQUEMENT les cartes de poker visibles dans cette image.

FORMAT OBLIGATOIRE pour chaque carte : [VALEUR][COULEUR]
- Valeurs : A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
- Couleurs : S, H, D, C

ZONES À ANALYSER :
1. Cartes personnelles (bas de l'écran, 2 cartes max)
2. Board/Flop/Turn/River (centre, 5 cartes max)

JSON DE RÉPONSE :
{
  "hero_cards": ["AS", "KH"] ou null,
  "community_cards": ["QD", "JC", "TS"] ou [],
  "detected_count": <nombre total de cartes détectées>,
  "confidence": <0.0 à 1.0>
}

Si tu n'es pas sûr à 100% d'une carte, utilise null. Précision > Rapidité."""

    @staticmethod
    def get_pot_and_blinds_prompt() -> str:
        """Prompt spécialisé pour pot et blinds"""
        return """MISSION : Détecte les montants (pot, blinds, mises) dans cette table de poker.

ÉLÉMENTS À CHERCHER :
1. Pot central (souvent au milieu avec des jetons)
2. Small Blind (SB)
3. Big Blind (BB)
4. Mises actuelles des joueurs

JSON DE RÉPONSE :
{
  "pot": <nombre> ou null,
  "small_blind": <nombre> ou null,
  "big_blind": <nombre> ou null,
  "player_bets": [<nombre>, <nombre>, <nombre>] ou []
}

Cherche des nombres affichés près des positions ou au centre de la table."""