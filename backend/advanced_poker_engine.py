"""
Moteur de poker avancé utilisant des librairies spécialisées
"""
import random
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import itertools

try:
    from treys import Card as TreysCard, Evaluator, Deck
    TREYS_AVAILABLE = True
except ImportError:
    TREYS_AVAILABLE = False
    print("Treys non disponible, utilisation du moteur basique")

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

@dataclass
class AnalysisResult:
    action: str
    confidence: float
    reasoning: str
    hand_strength: float
    pot_odds: float
    phase: str
    equity: float
    hand_type: str
    outs: int
    implied_odds: float
    position_factor: float
    icm_factor: float

class AdvancedPokerEngine:
    """Moteur de poker avancé avec calculs d'équité réels"""
    
    def __init__(self):
        self.evaluator = Evaluator() if TREYS_AVAILABLE else None
        
        # Tableaux de starting hands pour Spin & Go
        self.starting_hands = {
            'premium': [
                'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88',
                'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s',
                'AKo', 'AQo', 'AJo'
            ],
            'strong': [
                '77', '66', '55', '44', '33', '22',
                'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
                'ATo', 'A9o', 'A8o', 'A7o',
                'KQs', 'KJs', 'KTs', 'K9s', 'K8s',
                'KQo', 'KJo', 'KTo',
                'QJs', 'QTs', 'Q9s', 'QJo', 'QTo',
                'JTs', 'J9s', 'JTo', 'T9s', '98s', '87s', '76s'
            ],
            'playable': [
                'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
                'K9o', 'K8o', 'K7o', 'K6o',
                'Q9o', 'Q8o', 'J9o', 'J8o',
                'T8s', 'T7s', '97s', '86s', '75s', '65s', '54s',
                'T9o', '98o', '87o', '76o', '65o'
            ]
        }
        
        # Tableaux ICM pour Spin & Go 3 joueurs
        self.icm_adjustments = {
            'early': 1.0,      # Pas d'ajustement ICM
            'middle': 1.1,     # Légère prudence
            'late': 1.3,       # Forte prudence ICM
            'heads_up': 0.9    # Plus agressif en HU
        }
        
        # Ranges d'ouverture selon la position et phase
        self.opening_ranges = {
            'early': {
                Position.DEALER: 0.15,      # 15% des mains au BTN
                Position.SMALL_BLIND: 0.12, # 12% en SB
                Position.BIG_BLIND: 0.10    # 10% en BB (défense)
            },
            'middle': {
                Position.DEALER: 0.25,
                Position.SMALL_BLIND: 0.20,
                Position.BIG_BLIND: 0.15
            },
            'late': {
                Position.DEALER: 0.40,
                Position.SMALL_BLIND: 0.35,
                Position.BIG_BLIND: 0.25
            },
            'heads_up': {
                Position.DEALER: 0.70,      # Très large en HU BTN
                Position.SMALL_BLIND: 0.70,
                Position.BIG_BLIND: 0.50
            }
        }

    def convert_card_to_treys(self, card: Card) -> int:
        """Convertit une carte du format interne vers Treys"""
        if not TREYS_AVAILABLE:
            return None
            
        rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', 
                   '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
        suit_map = {'h': 'h', 'd': 'd', 'c': 'c', 's': 's'}
        
        return TreysCard.new(rank_map[card.rank] + suit_map[card.suit])

    def get_hand_string(self, cards: List[Card]) -> str:
        """Convertit une paire de cartes en string (ex: AKs, 72o)"""
        if len(cards) != 2:
            return "XX"
            
        rank1, rank2 = cards[0].rank, cards[1].rank
        suit1, suit2 = cards[0].suit, cards[1].suit
        
        # Ordre des rangs pour comparaison
        rank_order = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                     '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        # Mettre la carte la plus haute en premier
        if rank_order[rank1] < rank_order[rank2]:
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1
        
        # Ajouter 's' si suited, 'o' si offsuit
        if rank1 == rank2:  # Paire
            return f"{rank1}{rank2}"
        elif suit1 == suit2:  # Suited
            return f"{rank1}{rank2}s"
        else:  # Offsuit
            return f"{rank1}{rank2}o"

    def calculate_hand_equity(self, hero_cards: List[Card], community_cards: List[Card], 
                            num_opponents: int = 2) -> float:
        """Calcule l'équité réelle de la main avec Monte Carlo"""
        if not TREYS_AVAILABLE or len(hero_cards) != 2:
            return self.estimate_hand_strength_basic(hero_cards, community_cards)
        
        try:
            # Conversion vers Treys
            hero_treys = [self.convert_card_to_treys(card) for card in hero_cards]
            board_treys = [self.convert_card_to_treys(card) for card in community_cards]
            
            # Créer un deck sans les cartes connues
            deck = Deck()
            known_cards = hero_treys + board_treys
            for card in known_cards:
                if card in deck.cards:
                    deck.cards.remove(card)
            
            wins = 0
            ties = 0
            simulations = 1000  # Réduit pour la performance
            
            for _ in range(simulations):
                # Mélanger le deck
                random.shuffle(deck.cards)
                
                # Compléter le board si nécessaire
                simulation_board = board_treys.copy()
                cards_needed = 5 - len(community_cards)
                if cards_needed > 0:
                    simulation_board.extend(deck.cards[:cards_needed])
                
                # Générer les mains adverses
                remaining_cards = deck.cards[cards_needed:]
                hero_score = self.evaluator.evaluate(hero_treys, simulation_board)
                
                opponent_scores = []
                for i in range(num_opponents):
                    if len(remaining_cards) >= 2:
                        opponent_cards = remaining_cards[i*2:(i+1)*2]
                        opponent_score = self.evaluator.evaluate(opponent_cards, simulation_board)
                        opponent_scores.append(opponent_score)
                
                if opponent_scores:
                    best_opponent_score = min(opponent_scores)  # Plus bas = mieux dans Treys
                    if hero_score < best_opponent_score:
                        wins += 1
                    elif hero_score == best_opponent_score:
                        ties += 1
            
            equity = (wins + ties/2) / simulations
            return max(0.01, min(0.99, equity))  # Borner entre 1% et 99%
            
        except Exception as e:
            print(f"Erreur calcul équité: {e}")
            return self.estimate_hand_strength_basic(hero_cards, community_cards)

    def estimate_hand_strength_basic(self, hero_cards: List[Card], community_cards: List[Card]) -> float:
        """Estimation basique de la force de main sans librairie"""
        if not hero_cards or len(hero_cards) != 2:
            return 0.1
        
        hand_string = self.get_hand_string(hero_cards)
        
        # Classification basée sur les starting hands
        if hand_string in self.starting_hands['premium']:
            base_strength = 0.85
        elif hand_string in self.starting_hands['strong']:
            base_strength = 0.65
        elif hand_string in self.starting_hands['playable']:
            base_strength = 0.45
        else:
            base_strength = 0.25
        
        # Ajustements selon le board
        if community_cards:
            # Bonus si on a une paire avec le board
            hero_ranks = [card.rank for card in hero_cards]
            board_ranks = [card.rank for card in community_cards]
            
            if any(rank in board_ranks for rank in hero_ranks):
                base_strength += 0.15  # Bonus pour top pair, etc.
            
            # Bonus pour couleur
            hero_suits = [card.suit for card in hero_cards]
            if hero_suits[0] == hero_suits[1]:  # Hero suited
                board_suits = [card.suit for card in community_cards]
                same_suit_count = sum(1 for suit in board_suits if suit in hero_suits)
                if same_suit_count >= 2:  # Draw couleur
                    base_strength += 0.10
        
        return max(0.01, min(0.99, base_strength))

    def calculate_pot_odds(self, pot_size: int, bet_to_call: int) -> float:
        """Calcule les pot odds"""
        if bet_to_call <= 0:
            return float('inf')
        return pot_size / bet_to_call

    def calculate_outs(self, hero_cards: List[Card], community_cards: List[Card]) -> int:
        """Estime le nombre d'outs approximatif"""
        if not hero_cards or len(hero_cards) != 2:
            return 0
        
        outs = 0
        hero_ranks = [card.rank for card in hero_cards]
        hero_suits = [card.suit for card in hero_cards]
        
        if community_cards:
            board_ranks = [card.rank for card in community_cards]
            board_suits = [card.suit for card in community_cards]
            
            # Outs pour paire
            for rank in hero_ranks:
                if rank not in board_ranks:
                    outs += 3  # 3 cartes restantes de ce rang
            
            # Outs pour couleur (approximatif)
            if hero_suits[0] == hero_suits[1]:  # Hero suited
                same_suit_on_board = sum(1 for suit in board_suits if suit == hero_suits[0])
                if same_suit_on_board >= 2:
                    outs += max(0, 9 - same_suit_on_board)  # Outs couleur restants
            
            # Outs pour suite (approximatif)
            rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                          '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            
            all_ranks = hero_ranks + board_ranks
            rank_nums = sorted([rank_values[rank] for rank in all_ranks])
            
            # Check pour suite ouverte (approximatif)
            if len(set(rank_nums)) >= 4:
                gaps = [rank_nums[i+1] - rank_nums[i] for i in range(len(rank_nums)-1)]
                if any(gap == 1 for gap in gaps[:3]):  # Suite potentielle
                    outs += 4  # Approximation pour suite
        
        return min(outs, 20)  # Maximum 20 outs

    def determine_game_phase(self, game_state: GameState) -> str:
        """Détermine la phase de jeu basée sur les stacks et blinds"""
        active_players = [p for p in game_state.players if p.is_active]
        if len(active_players) <= 2:
            return 'heads_up'
        
        avg_stack = sum(p.stack for p in active_players) / len(active_players)
        bb_ratio = avg_stack / game_state.big_blind if game_state.big_blind > 0 else 50
        
        if bb_ratio > 20:
            return 'early'
        elif bb_ratio > 12:
            return 'middle'
        else:
            return 'late'

    def get_recommended_action(self, game_state: GameState, hero_position: Position, 
                             aggressiveness: float = 0.5) -> AnalysisResult:
        """Analyse avancée avec vraies librairies de poker"""
        
        # Trouver le héros
        hero = None
        for player in game_state.players:
            if player.position == hero_position:
                hero = player
                break
        
        if not hero or not hero.cards or len(hero.cards) != 2:
            return AnalysisResult(
                action='fold',
                confidence=0.1,
                reasoning='Cartes du héros non disponibles',
                hand_strength=0.0,
                pot_odds=0.0,
                phase='unknown',
                equity=0.0,
                hand_type='unknown',
                outs=0,
                implied_odds=0.0,
                position_factor=0.0,
                icm_factor=0.0
            )
        
        # Calculs avancés
        phase = self.determine_game_phase(game_state)
        hand_string = self.get_hand_string(hero.cards)
        equity = self.calculate_hand_equity(hero.cards, game_state.community_cards)
        outs = self.calculate_outs(hero.cards, game_state.community_cards)
        
        # Calcul des pot odds
        max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
        bet_to_call = max(0, max_bet - hero.current_bet)
        pot_odds = self.calculate_pot_odds(game_state.pot, bet_to_call) if bet_to_call > 0 else 0
        
        # Facteurs de position
        position_factor = self.get_position_factor(hero_position, phase)
        
        # Facteurs ICM
        icm_factor = self.icm_adjustments.get(phase, 1.0)
        
        # Odds implicites (simplifiées)
        implied_odds = pot_odds * 1.5 if pot_odds > 0 else 0
        
        # Logique de décision améliorée
        action, confidence, reasoning = self.make_decision(
            hand_string, equity, pot_odds, outs, phase, position_factor, 
            icm_factor, aggressiveness, bet_to_call, hero.stack
        )
        
        # Classification de la main
        hand_type = self.classify_hand_type(hero.cards, game_state.community_cards)
        
        return AnalysisResult(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            hand_strength=equity,
            pot_odds=pot_odds,
            phase=phase,
            equity=equity,
            hand_type=hand_type,
            outs=outs,
            implied_odds=implied_odds,
            position_factor=position_factor,
            icm_factor=icm_factor
        )

    def make_decision(self, hand_string: str, equity: float, pot_odds: float, outs: int,
                     phase: str, position_factor: float, icm_factor: float, 
                     aggressiveness: float, bet_to_call: int, stack: int) -> Tuple[str, float, str]:
        """Logique de décision avancée"""
        
        reasons = []
        
        # Décision basée sur l'équité et les pot odds
        if bet_to_call == 0:  # Pas de mise à suivre
            if equity >= 0.6:
                action = 'raise'
                confidence = 0.85
                reasons.append(f"Main forte ({equity:.1%}) - Value bet")
            elif equity >= 0.4:
                action = 'call' if phase == 'early' else 'raise'
                confidence = 0.70
                reasons.append(f"Main décente ({equity:.1%}) - Jouable")
            else:
                action = 'fold' if phase in ['early', 'middle'] else 'call'
                confidence = 0.60
                reasons.append(f"Main faible ({equity:.1%}) - Prudence requise")
        
        else:  # Il y a une mise à suivre
            # Calcul des odds requis
            pot_equity_needed = 1 / (pot_odds + 1) if pot_odds > 0 else 0.5
            
            if equity >= pot_equity_needed * 1.2:  # 20% de marge
                action = 'call'
                confidence = 0.80
                reasons.append(f"Équité ({equity:.1%}) > Odds requises ({pot_equity_needed:.1%})")
            elif equity >= pot_equity_needed and outs >= 8:
                action = 'call'
                confidence = 0.65
                reasons.append(f"Draw avec {outs} outs - Odds correctes")
            elif equity >= 0.7:
                action = 'raise'
                confidence = 0.90
                reasons.append(f"Main premium ({equity:.1%}) - Value raise")
            else:
                action = 'fold'
                confidence = 0.75
                reasons.append(f"Équité insuffisante ({equity:.1%}) vs odds ({pot_equity_needed:.1%})")
        
        # Ajustements selon la phase
        if phase == 'late' and action == 'fold' and equity >= 0.3:
            action = 'call'
            confidence = max(0.5, confidence - 0.2)
            reasons.append("Ajustement phase tardive - Plus large")
        
        if phase == 'heads_up' and action == 'fold' and equity >= 0.35:
            action = 'raise'
            confidence = max(0.6, confidence - 0.1)
            reasons.append("Heads-up - Jeu agressif requis")
        
        # Ajustements ICM
        if icm_factor > 1.1 and action == 'raise' and equity < 0.8:
            action = 'call'
            confidence = max(0.5, confidence - 0.15)
            reasons.append("Ajustement ICM - Plus conservateur")
        
        # Ajustements d'agressivité
        aggression_modifier = (aggressiveness - 0.5) * 0.2
        confidence = max(0.1, min(0.95, confidence + aggression_modifier))
        
        if aggressiveness > 0.7 and action == 'call' and equity >= 0.5:
            action = 'raise'
            reasons.append("Style agressif appliqué")
        elif aggressiveness < 0.3 and action == 'raise' and equity < 0.7:
            action = 'call'
            reasons.append("Style conservateur appliqué")
        
        reasoning = " | ".join(reasons)
        return action, confidence, reasoning

    def get_position_factor(self, position: Position, phase: str) -> float:
        """Facteur de position selon la phase"""
        base_factors = {
            Position.DEALER: 1.15,      # Avantage au bouton
            Position.SMALL_BLIND: 0.9,  # Désavantage SB
            Position.BIG_BLIND: 1.0     # Neutre BB
        }
        
        phase_multipliers = {
            'early': 1.0,
            'middle': 1.1,
            'late': 1.2,
            'heads_up': 1.3
        }
        
        base = base_factors.get(position, 1.0)
        multiplier = phase_multipliers.get(phase, 1.0)
        
        return base * multiplier

    def classify_hand_type(self, hero_cards: List[Card], community_cards: List[Card]) -> str:
        """Classifie le type de main"""
        if not community_cards:
            hand_string = self.get_hand_string(hero_cards)
            if hand_string in self.starting_hands['premium']:
                return 'Premium hand'
            elif hand_string in self.starting_hands['strong']:
                return 'Strong hand'
            elif hand_string in self.starting_hands['playable']:
                return 'Playable hand'
            else:
                return 'Marginal hand'
        
        # Avec community cards, analyser les combinaisons
        hero_ranks = [card.rank for card in hero_cards]
        board_ranks = [card.rank for card in community_cards]
        
        # Check pour paire
        if any(rank in board_ranks for rank in hero_ranks):
            return 'Paired hand'
        
        # Check pour draw
        hero_suits = [card.suit for card in hero_cards]
        if hero_suits[0] == hero_suits[1]:
            board_suits = [card.suit for card in community_cards]
            same_suit_count = sum(1 for suit in board_suits if suit == hero_suits[0])
            if same_suit_count >= 2:
                return 'Flush draw'
        
        return 'High card'

    def parse_cards_from_text(self, text: str) -> List[Card]:
        """Parse amélioré des cartes depuis le texte"""
        cards = []
        # Patterns plus robustes
        import re
        
        # Pattern pour capturer les cartes (AS, KH, QD, JC, TS, 9H, etc.)
        pattern = r'\b([AKQJT2-9])([HDCS])\b'
        matches = re.findall(pattern, text.upper())
        
        for rank, suit in matches:
            # Normalisation
            if rank == 'T':
                rank = 'T'  # 10 reste T
            suit = suit.lower()  # minuscule pour cohérence
            
            cards.append(Card(rank=rank, suit=suit))
        
        return cards