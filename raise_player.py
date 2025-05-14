from pypokerengine.players import BasePokerPlayer
import numpy as np
import time
import copy
import random

from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card


class RaisedPlayer(BasePokerPlayer):

  # how many simulations to run preflop vs. postflop
  PRE_FLOP_SIMS = 200
  POST_FLOP_SIMS = 300

  # adversarial search depth
  SEARCH_DEPTH = 2

  #TODO: unimplemented functionally
  # ms to leave as a safety margin (max total = 100ms)
  TIMEOUT_MS = 85

  # linear weights for evaluate()
  WEIGHTS = {
      'mc_winrate':  1.0, #entire score on montecarlo
      'pot_odds':    0.0,
      'hole_is_pair':0.0,
      'board_count': 0.0,
      # TODO add weights with features
    }
  
  STREETS = ['preflop', 'flop', 'turn', 'river', 'showdown']
  
  DECK = set(['S2', 'H2', 'D2', 'C2',
          'S3', 'H3', 'D3', 'C3',
          'S4', 'H4', 'D4', 'C4',
          'S5', 'H5', 'D5', 'C5',
          'S6', 'H6', 'D6', 'C6',
          'S7', 'H7', 'D7', 'C7',
          'S8', 'H8', 'D8', 'C8',
          'S9', 'H9', 'D9', 'C9',
          'ST', 'HT', 'DT', 'CT',
          'SJ', 'HJ', 'DJ', 'CJ',
          'SQ', 'HQ', 'DQ', 'CQ',
          'SK', 'HK', 'DK', 'CK',
          'SA', 'HA', 'DA', 'CA'])

    
  def declare_action(self, valid_actions, hole_card, round_state):
    """Called each turn to return the chosen move based on valid actions, hole cards, and round state."""
    start = time.time()
    
    #TODO:
    #1. implement time-checks throughout the code and fallback
    #2. optimize everything heavily (map, vectorization, parallelization)
    #3. log static features, action taken, and game result (win/loss amount of money) 
    #   quantify and output to .csv
    #4. determine optimal ordering of actions to improve alpha-beta pruning
    
    
    #program flow:
    #1. simplify state to abstract state and log static features for representation
    #2. run minimax on starting on this state
    #3  minimax simulates all possible actions (only 3!) and their outcomes while alternating players
    #4. leaf nodes are evaluated with eval_state()
    #5.     eval_state() performs montecarlo simulations to estimate winrate
    #6.     eval_state() calls evaluate_heuristic() to compute a score based on features (which now include montecarlo winrate)
    #7.     eval_heuristic() computes a score based on features and (learned) weights
    #8. eval_state() returns the score to minimax, score used to evaluate actions
    
    
    #potentially more utilities in the poker engine like 
    # automated montecarlo, valid action checker, ...
    #TODO: look into this

    abstract_state, static_features = self.preprocess(round_state, hole_card, valid_actions)

    
    best_action = self.fallback(valid_actions)

     # Iterate and search until timeout
    alpha, beta = -np.inf, np.inf
    depth = self.SEARCH_DEPTH
    for action_dict in valid_actions:

      # Timeout check
      if (time.time() - start) * 1000 > self.TIMEOUT_MS:
        break
      # Simulate and score
      next_state = self.simulate_action(abstract_state, action_dict)
      val = self.min_node(next_state, depth-1, alpha, beta, start)
      if val > alpha:
        alpha = val
        best_action = action_dict
        # Return only the action string
    return best_action['action']


  def fallback(self, valid_actions):

    for a in valid_actions:
      if a['action'] == 'call':
        return a
      
    for a in valid_actions:
      if a['action'] == 'fold':
        return a
      
    return valid_actions[0]

  def receive_game_start_message(self, game_info): # dont change
    """Called at game start to receive global information such as player details and chip counts."""
    pass

  def receive_round_start_message(self, round_count, hole_card, seats): #dont change
    """Called at the beginning of each round to provide hole cards and seating details, 
    and to reset round-specific data."""
    pass

  def receive_street_start_message(self, street, round_state): #dont change
    """ Called at the start of each betting round (street) 
    to update the game state with new community cards and betting context."""
    pass

  def receive_game_update_message(self, action, round_state): #dont change
    """ Called during rounds to update the agent on the actions 
    taken by players and current game developments."""
    pass

  def receive_round_result_message(self, winners, hand_info, round_state): #dont change
    """Called at the end of a round to deliver the results, 
    including winners, hand details, and final game state"""
    pass

  
  def preprocess(self, round_state, hole_card, valid_actions):
    """Generates a simplified state representation for simulation and static features for training"""    
    #street and street index
    street = round_state['street']
    street_idx = self.STREETS.index(street)
    
    #by default, these are the roles
    is_playing = True
    has_folded = False
    opponent_folded = False
    
    #tracks last two moves for street state updates in simulations
    last_moves = []
    street_actions = round_state['action_histories'].get(street, [])

    if len(street_actions) > 0:
      last_moves.append(street_actions[-1]['action'].lower())
      if len(street_actions) > 1:
        last_moves.append(street_actions[-2]['action'].lower())
      
    #determines pot amount and current bets// no sidepots in 2 player games
    pot = round_state['pot']['main']['amount']
    my_bet, opp_bet = self.get_bet_amounts(round_state)
    
    #abstract state attributes - could be optimized for space
    abstract_state = {
      'my_cards': hole_card,
      'community_cards': round_state['community_card'],
      'my_bet': my_bet,
      'opp_bet': opp_bet,
      'street': street,
      'street_idx': street_idx,
      'pot': pot,
      'is_playing': is_playing,
      'valid_actions': valid_actions,
      'last_moves': last_moves,
      'raise_count': 0,
      'features': self.extract_static_features(round_state, hole_card, my_bet, opp_bet, pot, street)
    }

    
    
    return abstract_state, abstract_state['features']
  
  def extract_static_features(self, round_state, hole_card, my_bet, opp_bet, pot, street):
    """Designs features for evaluating position strength and making decisions."""
    
    feats = {}
    # Pot odds feature (if calling)
    to_call = opp_bet - my_bet
    feats['pot_odds'] = to_call / pot if pot > 0 else 0.0
    # Add more static features (hand strength, pairs, etc.)
    return feats

  
  def get_bet_amounts(self, round_state):
    """iterates through action histories and sums up bet amounts
                  (((OPTIMIZE ME)))
    """
    
    #optimize via dynamic programming methods pls
    my_bet = 0
    opp_bet = 0
    
    for street, acts in round_state['action_histories'].items():
      for turn in acts:
        if turn['uuid'] == self.uuid:
          my_bet += turn.get('amount', 0)
        else:
          opp_bet += turn.get('amount', 0)
    return my_bet, opp_bet
    

  
  def min_node(self, state, depth, alpha, beta, start_time):
    
    if depth == 0 or self.is_terminal_state(state):# or time.time() - start_time > self.TIMEOUT_MS / 1000:
      return self.eval_state(state, start_time)
    
    if (time.time() - start_time) * 1000 > self.TIMEOUT_MS:
      return self.eval_state(state, start_time)
    
    val = np.inf
    
    for possible_action_dict in self.filter_actions(state):

      next_state = self.simulate_action(state, possible_action_dict)
      val = min(val, self.max_node(next_state, depth-1, alpha, beta, start_time))
      beta = min(beta, val)
      #pruning
      if beta <= alpha:
        break
    return val
  
  
  def max_node(self, state, depth, alpha, beta, start_time):
    
    if depth == 0 or self.is_terminal_state(state):# or time.time() - start_time > self.TIMEOUT_MS / 1000:
      return self.eval_state(state, start_time)
    if (time.time() - start_time) * 1000 > self.TIMEOUT_MS:
      return self.eval_state(state, start_time)
    
    val = -np.inf
    
    for possible_action_dict in self.filter_actions(state):
      next_state = self.simulate_action(state, possible_action_dict)
      val = max(val, self.min_node(next_state, depth - 1, alpha, beta, start_time))
      alpha = max(alpha, val)
      # pruning
      if alpha >= beta:
        break
    return val


  def filter_actions(self, state):
    # cap raises at 4 per street
    acts = []
    for a in state['valid_actions']:
      if a['action'] == 'raise' and state['raise_count'] >= 4:
        continue
      acts.append(a)
    return acts
  
  def simulate_action(self, state, action_dict):

    new_state = copy.deepcopy(state)

    # whose turn it was
    is_acting = new_state['is_playing']
    act       = action_dict['action']
    amt       = action_dict.get('amount', 0)

    if act == 'call':
      # add call amt to pot and to the proper player's bet
      new_state['pot'] += amt
      if is_acting:
        new_state['my_bet']  += amt
      else:
        new_state['opp_bet'] += amt

    elif act == 'raise':
      # fixed‑limit: exactly $10 above opponent’s bet
      to_call   = abs(new_state['opp_bet'] - new_state['my_bet'])
      # total invests = call + raise
      new_state['pot']    += to_call + 10
      
      if is_acting:
        new_state['my_bet']  += to_call + 10
      else:
        new_state['opp_bet'] += to_call + 10
      new_state['raise_count'] += 1

    else:  # fold
      if is_acting:
        new_state['has_folded']     = True
      else:
        new_state['opponent_folded'] = True

    # switch whose turn it is
    new_state['is_playing'] = not is_acting

    # street transition
    if new_state['my_bet'] == new_state['opp_bet'] and act != 'fold':
      idx = new_state['street_idx']

      if idx < len(self.STREETS)-1:
        new_state['street_idx'] += 1
        new_state['street'] = self.STREETS[new_state['street_idx']]
        known = [c for c in new_state['community_cards'] if c != '_']

        if new_state['street'] == 'flop':
          new_state['community_cards'] = known + ['_','_','_']
          
        elif new_state['street'] in ['turn','river']:
          new_state['community_cards'] = known + ['_']

        new_state['last_moves'] = []
        new_state['raise_count'] = 0
    
    return new_state


  def montecarlo(self, state):
    """Estimate winrate from the current (abstract/simple) state by simulating random outcomes."""
    
    wins = 0
    my_cards = state['my_cards']
    community = [c for c in state['community_cards'] if c != '_']

    sims = self.PRE_FLOP_SIMS if len(community) == 0 else self.POST_FLOP_SIMS
    deck = list(self.DECK - set(my_cards) - set(community))

    for _ in range(sims):
      sample = deck.copy()
      opp_cards = random.sample(sample, 2)
      for card in opp_cards:
          sample.remove(card)


      needed = 5 - len(community)

      board = community + random.sample(sample, needed)
      my_score = HandEvaluator.eval_hand([Card.from_str(card) for card in my_cards],[Card.from_str(card) for card in board])
      opp_score = HandEvaluator.eval_hand([Card.from_str(card) for card in opp_cards],[Card.from_str(card) for card in board])

      if my_score > opp_score:
        wins += 1
      elif my_score == opp_score:
        wins += 0.5

    return wins / sims
    
  def eval_state(self, state, start_time):
    """Determines the scores at leaf nodes of minimax tree. Performs montecarlo simulations
    to estimate winrate and use for action evaluation."""

    if state.get('has_folded'):
      return -state['pot']
    if state.get('opponent_folded'):
      return state['pot']

    # montecarlo simulation as a feature
    winrate = self.montecarlo(state)
    
    estimated_features = state.get('features', {}).copy()
    estimated_features['mc_winrate'] = winrate
    
    return sum(self.WEIGHTS.get(k,0)*v for k,v in estimated_features.items())
  

  def is_terminal_state(self, state):
    return state.get('has_folded') or state.get('opponent_folded') or state['street'] == 'showdown'

  def evaluate_heuristic(self, features):
    """Compute a heuristic score for the current game state 
    using features generated in preprocessing
    Combines factors such as hand strength, pot odds, and opponent behavior 
    through weighted evaluation to guide the final decision."""
    
    #TODO: implement nonlinear scoring function via network
    score = 0.0
    for k,w in self.WEIGHTS.items():
      if k in features:
        score += w * features.get(k, 0)
    return score
  
  def update_opponent_model(self, round_state):
    """Update the model of the opponent's playing style and tendencies
    based on their actions and game history.
    This can be used to adjust the strategy dynamically."""

    # if last and last['uuid'] != self.uuid: 
    pass


def setup_ai():
  #return RandomPlayer()
  return RaisedPlayer()
