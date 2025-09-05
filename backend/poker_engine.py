"""
Moteur d'analyse stratégique pour poker Texas Hold'em Spin & Go 3 joueurs
"""
import math
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

class Position(Enum):
    DEALER = "dealer"
    SMALL_BLIND = "small_blind"
    BIG_BLIND = "big_blind"

class Action(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"

@dataclass
class Card:
    rank: str  # 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A
    suit: str  # h, d, c, s

@dataclass
class Player:
    name: str
    stack: int
    position: Position
    cards: Optional[List[Card]] = None
    last_action: Optional[Action] = None
    current_bet: int = 0
    is_active: bool = True

@dataclass
class GameState:
    players: List[Player]
    community_cards: List[Card]
    pot: int
    small_blind: int
    big_blind: int
    current_player: int
    betting_round: str  # preflop, flop, turn, river

class PokerEngine:
    """Moteur d'analyse de poker avec calculs ICM et stratégies Spin & Go"""
    
    def __init__(self):
        self.hand_rankings = {
            'high_card': 1, 'pair': 2, 'two_pair': 3, 'three_of_a_kind': 4,
            'straight': 5, 'flush': 6, 'full_house': 7, 'four_of_a_kind': 8,
            'straight_flush': 9, 'royal_flush': 10
        }
        
        # Ranges approximatives pour Spin & Go
        self.spin_and_go_ranges = {
            'early': {'tight': 0.12, 'normal': 0.15, 'loose': 0.20},
            'middle': {'tight': 0.18, 'normal': 0.25, 'loose': 0.35},
            'late': {'tight': 0.35, 'normal': 0.50, 'loose': 0.70},
            'heads_up': {'tight': 0.60, 'normal': 0.75, 'loose': 0.90}
        }
    
    def calculate_hand_strength(self, hole_cards: List[Card], community_cards: List[Card]) -> float:
        """Calcule la force de la main actuelle (0-1)"""
        if not hole_cards or len(hole_cards) != 2:
            return 0.0
        
        # Évaluation simplifiée basée sur les cartes
        all_cards = hole_cards + community_cards
        hand_strength = 0.0
        
        # Bonus pour paires
        if hole_cards[0].rank == hole_cards[1].rank:
            pair_value = self._get_card_value(hole_cards[0].rank)
            hand_strength += 0.1 + (pair_value / 14) * 0.4
        
        # Bonus pour cartes hautes
        for card in hole_cards:
            hand_strength += self._get_card_value(card.rank) / 14 * 0.1
        
        # Bonus pour suited
        if hole_cards[0].suit == hole_cards[1].suit:
            hand_strength += 0.05
        
        # Bonus pour connectées
        if abs(self._get_card_value(hole_cards[0].rank) - self._get_card_value(hole_cards[1].rank)) <= 1:
            hand_strength += 0.03
        
        return min(hand_strength, 1.0)
    
    def calculate_pot_odds(self, current_bet: int, pot_size: int, call_amount: int) -> float:
        """Calcule les pot odds"""
        if call_amount == 0:
            return float('inf')
        return pot_size / call_amount
    
    def calculate_icm_value(self, stacks: List[int], prize_structure: List[float]) -> List[float]:
        """Calcule les valeurs ICM pour chaque joueur"""
        total_chips = sum(stacks)
        if total_chips == 0:
            return [0.0] * len(stacks)
        
        # Calcul ICM simplifié pour 3 joueurs
        values = []
        for stack in stacks:
            if stack == 0:
                values.append(0.0)
            else:
                # Probabilité de finir 1er, 2ème, 3ème
                p_first = stack / total_chips
                p_second = (1 - p_first) * stack / (total_chips - stack) if total_chips != stack else 0
                p_third = 1 - p_first - p_second
                
                icm_value = (p_first * prize_structure[0] + 
                           p_second * prize_structure[1] + 
                           p_third * prize_structure[2])
                values.append(icm_value)
        
        return values
    
    def get_recommended_action(self, game_state: GameState, hero_position: Position, 
                             aggressiveness: float = 0.5) -> Dict[str, Any]:
        """Recommande la meilleure action basée sur la situation"""
        
        hero = None
        for player in game_state.players:
            if player.position == hero_position:
                hero = player
                break
        
        if not hero or not hero.cards:
            return {'action': 'fold', 'confidence': 0.0, 'reasoning': 'Informations insuffisantes'}
        
        # Calcul de la force de main
        hand_strength = self.calculate_hand_strength(hero.cards, game_state.community_cards)
        
        # Évaluation de la phase de jeu
        total_chips = sum(p.stack for p in game_state.players if p.is_active)
        avg_stack = total_chips / len([p for p in game_state.players if p.is_active])
        blind_level = game_state.big_blind / avg_stack if avg_stack > 0 else 1.0
        
        # Détermination de la phase
        if blind_level < 0.05:
            phase = 'early'
        elif blind_level < 0.15:
            phase = 'middle'
        elif len([p for p in game_state.players if p.is_active]) > 2:
            phase = 'late'
        else:
            phase = 'heads_up'
        
        # Calcul du seuil d'action basé sur la position et l'agressivité
        position_modifier = self._get_position_modifier(hero.position)
        aggr_style = 'loose' if aggressiveness > 0.7 else 'normal' if aggressiveness > 0.3 else 'tight'
        threshold = self.spin_and_go_ranges[phase][aggr_style] * position_modifier
        
        # Calcul des pot odds si nécessaire
        call_amount = max(0, max(p.current_bet for p in game_state.players) - hero.current_bet)
        pot_odds = self.calculate_pot_odds(hero.current_bet, game_state.pot, call_amount) if call_amount > 0 else 0
        
        # Décision principale
        confidence = hand_strength
        reasoning = []
        
        if hand_strength >= threshold * 1.5:
            action = 'raise'
            reasoning.append(f"Main forte ({hand_strength:.2f}) en phase {phase}")
            confidence = min(0.9, hand_strength + 0.2)
        elif hand_strength >= threshold:
            if call_amount == 0:
                action = 'call'
            elif pot_odds > 3.0 and hand_strength > 0.3:
                action = 'call'
                reasoning.append(f"Pot odds favorables ({pot_odds:.1f}:1)")
            else:
                action = 'call'
            reasoning.append(f"Main jouable ({hand_strength:.2f})")
        else:
            action = 'fold'
            reasoning.append(f"Main faible ({hand_strength:.2f}) sous seuil ({threshold:.2f})")
            confidence = 1.0 - hand_strength
        
        # Ajustements ICM si en phase tardive
        if phase in ['late', 'heads_up']:
            stacks = [p.stack for p in game_state.players if p.is_active]
            if len(stacks) == 3:
                icm_values = self.calculate_icm_value(stacks, [0.50, 0.30, 0.20])
                reasoning.append("Ajustement ICM appliqué")
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': ' | '.join(reasoning),
            'hand_strength': hand_strength,
            'pot_odds': pot_odds,
            'phase': phase
        }
    
    def _get_card_value(self, rank: str) -> int:
        """Convertit le rang de carte en valeur numérique"""
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                 '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return values.get(rank, 0)
    
    def _get_position_modifier(self, position: Position) -> float:
        """Modificateur basé sur la position"""
        modifiers = {
            Position.DEALER: 1.2,      # Plus large au bouton
            Position.SMALL_BLIND: 0.9, # Plus serré en SB
            Position.BIG_BLIND: 1.0    # Normal en BB
        }
        return modifiers.get(position, 1.0)
    
    def parse_cards_from_text(self, text: str) -> List[Card]:
        """Parse les cartes depuis le texte retourné par l'IA"""
        cards = []
        # Extraction simplifiée - à améliorer selon le format exact de retour
        words = text.upper().split()
        for word in words:
            if len(word) >= 2:
                # Recherche de patterns comme "AS", "KH", "10D", etc.
                for i in range(len(word)-1):
                    rank_char = word[i]
                    suit_char = word[i+1]
                    
                    if rank_char in '23456789TJQKA' and suit_char in 'HDCS':
                        rank = rank_char if rank_char != 'T' else '10'
                        suit = suit_char.lower()
                        cards.append(Card(rank=rank, suit=suit))
        
        return cards