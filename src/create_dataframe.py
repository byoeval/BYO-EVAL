import json
import pandas as pd
from typing import List, Dict, Any, Set, Optional
from pathlib import Path

class AnnotationDataset:
    def __init__(self, annotation_dir: str, game: str = "chess"):
        """
        Initialize the dataset with the path to the annotation directory.
        
        Args:
            annotation_dir (str): Path to the directory containing JSON annotation files
            game (str): Type of game being analyzed, defaults to "chess"
        """
        self.annotation_dir = Path(annotation_dir)
        self.data = None
        self.game = game
        
    def load_annotations(self) -> None:
        """Load all JSON annotation files from the directory into a pandas DataFrame."""
        data_rows = []

        # Iterate through all JSON files in the annotation directory
        for json_file in self.annotation_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                annotation = json.load(f)
                
                if self.game == "chess":
                    # Extract board information
                    board = annotation.get('board', {})
                    board_info = {
                        'board_rows': board.get('dimensions', {}).get('rows'),
                        'board_columns': board.get('dimensions', {}).get('columns'),
                        'board_pattern': board.get('pattern'),
                        'board_location': board.get('location')
                    }

                    # Extract camera information
                    camera_info = {
                        'distance': board.get('camera', {}).get('final_distance'),
                        'angle': board.get('camera', {}).get('final_angle'),
                        'horizontal_angle': board.get('camera', {}).get('final_horizontal_angle')
                    }

                    # Extract pieces information
                    pieces = annotation.get('pieces', {})
                    pieces_info = []
                    piece_types = set()
                    piece_colors = set()
                    
                    for piece_id, piece_data in pieces.items():
                        piece_type = piece_data.get('type')
                        piece_color = piece_data.get('color')
                        
                        if piece_type:
                            piece_types.add(piece_type)
                        if piece_color:
                            piece_colors.add(piece_color)
                            
                        piece_info = {
                            'piece_id': piece_id,
                            'piece_type': piece_type,
                            'board_position': piece_data.get('board_position'),
                            'color': piece_color,
                            'scale': piece_data.get('scale'),
                            'image_position': piece_data.get('image_position'),
                            'random_rotation': piece_data.get('random_rotation'),
                            'max_rotation_angle': piece_data.get('max_rotation_angle', None)
                        }
                        pieces_info.append(piece_info)

                    # Extract noise information
                    noise = annotation.get('noise', {})
                    noise_info = {
                        'table_texture': noise.get('table_texture', {}).get('table_texture'),
                        'blur': noise.get('blur', {}).get('blur')
                    }

                    # Create a row for the DataFrame
                    row = {
                        'image': json_file.stem,
                        'game': self.game,
                        'camera': camera_info,
                        'noise': noise_info,
                        'setup': {}, # Empty dict for now, can be filled based on specific requirements
                        'board': board_info,
                        'pieces': pieces_info,
                        'piece_types': piece_types,
                        'piece_number': len(pieces_info),
                        'piece_colors': piece_colors
                    }
                    
                elif self.game == "poker":
                    # Extract scene setup information
                    scene_setup = annotation.get('scene_setup', {})
                    
                    # Extract camera information
                    camera_info = scene_setup.get('camera', {})
                    
                    # Extract lighting and table information for setup
                    setup_info = {
                        'lighting': scene_setup.get('lighting', {}).get('lighting'),
                        'table': scene_setup.get('table', {}),
                        'resolution': scene_setup.get('render', {}).get('resolution', {})
                    }
                    
                    # Extract noise information (may need to be adjusted based on actual JSON structure)
                    noise_info = {
                        'table_texture': scene_setup.get('table', {}).get('felt_color'),
                        'blur': annotation.get('noise', {}).get('blur', None)
                    }
                    
                    # Process card_overlap_layout if present
                    card_overlap_layout = annotation.get('card_overlap_layout', None)
                    card_overlap_info = {}
                    card_overlap_count = 0
                    
                    if card_overlap_layout:
                        card_overlap_info = {
                            'layout_id': card_overlap_layout.get('layout_id'),
                            'layout_mode': card_overlap_layout.get('layout_mode'),
                            'overall_cards': card_overlap_layout.get('overall_cards', 0),
                            'n_lines': card_overlap_layout.get('n_lines', 0),
                            'n_columns': card_overlap_layout.get('n_columns', 0),
                            'card_type_config': card_overlap_layout.get('card_type_config', {}),
                            'center_location': card_overlap_layout.get('center_location'),
                            'scale': card_overlap_layout.get('scale'),
                            'horizontal_overlap_factor': card_overlap_layout.get('horizontal_overlap_factor', 0),
                            'vertical_overlap_factor': card_overlap_layout.get('vertical_overlap_factor', 0),
                            'line_gap': card_overlap_layout.get('line_gap', 0),
                            'column_gap': card_overlap_layout.get('column_gap', 0),
                            'n_verso': card_overlap_layout.get('n_verso', 0),
                            'verso_loc': card_overlap_layout.get('verso_loc')
                        }
                        card_overlap_count = card_overlap_layout.get('overall_cards', 0)
                    
                    # Process community cards - handle case where community_cards is None
                    community_cards = annotation.get('community_cards', {})
                    if community_cards is None:
                        if card_overlap_layout is None:
                            print("Warning: community_cards is None, using empty dict instead")
                        community_cards = {}
                        
                    community_card_names = community_cards.get('card_names', [])
                    community_n_cards = community_cards.get('n_cards', 0)
                    community_location = community_cards.get('start_location')
                    community_scale = community_cards.get('scale')
                    community_n_verso = community_cards.get('n_verso', 0)
                    
                    # Calculate community card spread if location exists
                    community_horizontal_spread = 0.0
                    community_vertical_spread = 0.0
                    
                    if community_location and isinstance(community_location, list) and len(community_location) >= 2:
                        # Extract card gap information to calculate spread
                        card_gap = community_cards.get('card_gap', {})
                        base_gap_x = card_gap.get('base_gap_x', 0.0) if card_gap else 0.0
                        base_gap_y = card_gap.get('base_gap_y', 0.0) if card_gap else 0.0
                        
                        # Calculate the spread based on number of cards and gaps
                        if community_n_cards > 1:
                            community_horizontal_spread = base_gap_x * (community_n_cards - 1)
                            community_vertical_spread = base_gap_y * (community_n_cards - 1)
                    
                    community_info = {
                        'cards': community_card_names,
                        'n_cards': community_n_cards,
                        'location': community_location,
                        'scale': community_scale,
                        'n_verso': community_n_verso,
                        'horizontal_spread': community_horizontal_spread,
                        'vertical_spread': community_vertical_spread
                    }
                    
                    # Process players - handle case where players is None or empty
                    players = annotation.get('players', [])
                    if players is None:
                        print("Warning: players is None, using empty list instead")
                        players = []
                        
                    players_info = []
                    all_card_info = []
                    card_types = set()
                    
                    # Variables to track player positions for spread calculation
                    player_positions = []
                    total_piles = 0
                    
                    for player_idx, player in enumerate(players):
                        player_id = player.get('player_id')
                        hand_config = player.get('hand_config', {})
                        chip_area_config = player.get('chip_area_config', {})
                        
                        # Extract card information
                        card_names = hand_config.get('card_names', []) if hand_config else []
                        n_cards = hand_config.get('n_cards', 0) if hand_config else 0
                        location = hand_config.get('location') if hand_config else None
                        scale = hand_config.get('scale') if hand_config else None
                        n_verso = hand_config.get('n_verso', 0) if hand_config else 0
                        
                        # Store player position for spread calculation
                        if location and isinstance(location, list) and len(location) >= 2:
                            player_positions.append((location[0], location[1]))
                        
                        # Extract chip pile information
                        resolved_piles = chip_area_config.get('resolved_piles', []) if chip_area_config else []
                        piles_info = []
                        
                        # Count piles for this player
                        player_pile_count = len(resolved_piles)
                        total_piles += player_pile_count
                        
                        for pile in resolved_piles:
                            pile_info = {
                                'n_chips': pile.get('n_chips'),
                                'color': pile.get('color'),
                                'scale': pile.get('scale')
                            }
                            piles_info.append(pile_info)
                        
                        # Add card types to the set
                        for card in card_names:
                            if card:
                                card_types.add(card)
                        
                        # Create card info entries for this player
                        for i, card_name in enumerate(card_names):
                            if card_name:
                                card_info = {
                                    'card_id': f"{player_id}_card_{i}",
                                    'card_type': card_name,
                                    'owner': player_id,
                                    'is_community': False,
                                    'location': location,
                                    'scale': scale
                                }
                                all_card_info.append(card_info)
                        
                        player_info = {
                            'player_id': player_id,
                            'location': location,
                            'cards': card_names,
                            'n_cards': n_cards,
                            'n_verso': n_verso,
                            'piles': piles_info,
                            'n_piles': player_pile_count
                        }
                        players_info.append(player_info)
                    
                    # Add community cards to card_info
                    for i, card_name in enumerate(community_card_names):
                        if card_name:
                            card_types.add(card_name)
                            card_info = {
                                'card_id': f"community_card_{i}",
                                'card_type': card_name,
                                'owner': 'community',
                                'is_community': True,
                                'location': community_location,
                                'scale': community_scale
                            }
                            all_card_info.append(card_info)
                    
                    # Calculate player horizontal and vertical spread
                    player_horizontal_spread = 0.0
                    player_vertical_spread = 0.0
                    
                    if len(player_positions) >= 2:
                        # Find min and max coordinates
                        min_x = min(pos[0] for pos in player_positions)
                        max_x = max(pos[0] for pos in player_positions)
                        min_y = min(pos[1] for pos in player_positions)
                        max_y = max(pos[1] for pos in player_positions)
                        
                        # Calculate spread as the difference between max and min
                        player_horizontal_spread = max_x - min_x
                        player_vertical_spread = max_y - min_y
                    
                    # Calculate total card count including overlap layout
                    total_card_count = len(all_card_info) + card_overlap_count
                    
                    # Create a row for the DataFrame
                    row = {
                        'image': json_file.stem,
                        'game': self.game,
                        'camera': camera_info,
                        'noise': noise_info,
                        'setup': setup_info,
                        'card_info': all_card_info,
                        'card_types': card_types,
                        'card_number': len(all_card_info),
                        'players_info': players_info,
                        'player_number': len(players_info),
                        'community_info': community_info,
                        'card_overlap_layout': card_overlap_info,
                        'card_overlap_count': card_overlap_count,
                        # Add new columns for easier filtering
                        'n_cards': total_card_count,
                        'n_piles': total_piles,
                        'n_players': len(players_info),
                        'player_horizontal_spread': player_horizontal_spread,
                        'player_vertical_spread': player_vertical_spread,
                        'community_horizontal_spread': community_horizontal_spread,
                        'community_vertical_spread': community_vertical_spread,
                        'has_overlap_layout': card_overlap_layout is not None
                    }

                
                else:
                    # Default case or other game types
                    row = {
                        'image': json_file.stem,
                        'game': self.game
                    }
                
                data_rows.append(row)

        # Create DataFrame
        self.data = pd.DataFrame(data_rows)

        
    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all annotation data
        """
        if self.data is None:
            self.load_annotations()
        return self.data
    
    def get_pieces_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame where each row represents an individual piece from the annotations.
        
        Returns:
            pd.DataFrame: DataFrame with one row per piece
        """
        if self.data is None:
            self.load_annotations()
            
        if self.game != "chess":
            raise ValueError(f"get_pieces_dataframe is only available for chess games, not {self.game}")
            
        piece_rows = []
        for _, row in self.data.iterrows():
            for piece in row['pieces']:
                piece_row = {
                    'image': row['image'],
                    'game': self.game,
                    'piece_id': piece['piece_id'],
                    'piece_type': piece['piece_type'],
                    'board_position': piece['board_position'],
                    'color': piece['color'],
                    'scale': piece['scale'],
                    'image_position': piece['image_position'],
                    'random_rotation': piece['random_rotation'],
                    'max_rotation_angle': piece.get('max_rotation_angle')
                }
                piece_rows.append(piece_row)
                
        return pd.DataFrame(piece_rows)
    
    def get_cards_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame where each row represents an individual card from the poker annotations.
        
        Returns:
            pd.DataFrame: DataFrame with one row per card
        """
        if self.data is None:
            self.load_annotations()
            
        if self.game != "poker":
            raise ValueError(f"get_cards_dataframe is only available for poker games, not {self.game}")
            
        card_rows = []
        for _, row in self.data.iterrows():
            for card in row['card_info']:
                card_row = {
                    'image': row['image'],
                    'game': self.game,
                    'card_id': card['card_id'],
                    'card_type': card['card_type'],
                    'owner': card['owner'],
                    'is_community': card['is_community'],
                    'location': card['location'],
                    'scale': card['scale']
                }
                card_rows.append(card_row)
                
        return pd.DataFrame(card_rows)
    
    def get_players_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame where each row represents a player from the poker annotations.
        
        Returns:
            pd.DataFrame: DataFrame with one row per player
        """
        if self.data is None:
            self.load_annotations()
            
        if self.game != "poker":
            raise ValueError(f"get_players_dataframe is only available for poker games, not {self.game}")
            
        player_rows = []
        for _, row in self.data.iterrows():
            for player in row['players_info']:
                player_row = {
                    'image': row['image'],
                    'game': self.game,
                    'player_id': player['player_id'],
                    'location': player['location'],
                    'n_cards': player['n_cards'],
                    'n_verso': player['n_verso'],
                    'cards': player['cards'],
                    'piles': player['piles']
                }
                player_rows.append(player_row)
                
        return pd.DataFrame(player_rows)
    
    def get_card_overlap_layouts_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame where each row represents a card overlap layout from the poker annotations.
        
        Returns:
            pd.DataFrame: DataFrame with card overlap layout information
        """
        if self.data is None:
            self.load_annotations()
            
        if self.game != "poker":
            raise ValueError(f"get_card_overlap_layouts_dataframe is only available for poker games, not {self.game}")
            
        layout_rows = []
        for _, row in self.data.iterrows():
            if row['has_overlap_layout']:
                layout = row['card_overlap_layout']
                card_type_config = layout.get('card_type_config', {})
                
                # Extract center location components if available
                center_x, center_y, center_z = None, None, None
                if layout.get('center_location') and len(layout['center_location']) >= 3:
                    center_x = layout['center_location'][0]
                    center_y = layout['center_location'][1]
                    center_z = layout['center_location'][2]
                
                layout_row = {
                    'image': row['image'],
                    'game': self.game,
                    'layout_id': layout.get('layout_id'),
                    'layout_mode': layout.get('layout_mode'),
                    'overall_cards': layout.get('overall_cards', 0),
                    'n_lines': layout.get('n_lines', 0),
                    'n_columns': layout.get('n_columns', 0),
                    'scale': layout.get('scale'),
                    'center_x': center_x,
                    'center_y': center_y,
                    'center_z': center_z,
                    'horizontal_overlap_factor': layout.get('horizontal_overlap_factor', 0),
                    'vertical_overlap_factor': layout.get('vertical_overlap_factor', 0),
                    'line_gap': layout.get('line_gap', 0),
                    'column_gap': layout.get('column_gap', 0),
                    'n_verso': layout.get('n_verso', 0),
                    'verso_loc': layout.get('verso_loc'),
                    'card_type_mode': card_type_config.get('mode'),
                    'allow_repetition': card_type_config.get('allow_repetition', False)
                }
                layout_rows.append(layout_row)
                
        return pd.DataFrame(layout_rows)




if __name__ == "__main__":
    """
    Test script to verify the AnnotationDataset class with poker game data.
    """
    import os
    import sys
    import traceback
    from pathlib import Path
    from pprint import pprint
    
    # Set up file paths
    poker_json_path = "/home/cards_chip_standard/legend_json"
    test_file = "poker_chip_variations_img_00000_legend.json"
    
    print("\n========== TEST ANNOTATION DATASET ==========")
    print(f"Testing path: {poker_json_path}")
    
    try:
        # Create the dataset with game type "poker"
        print("\nInitializing AnnotationDataset with game=poker...")
        poker_dataset = AnnotationDataset(poker_json_path, game="poker")
        
        # Load and print basic dataset information
        print("\n===== Loading Poker Dataset =====")
        df = poker_dataset.get_dataframe()
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        if df.empty:
            print("\nWARNING: DataFrame is empty. No files were processed successfully.")
            sys.exit(1)
            
        # Print some key information about the first row
        print("\n===== First Row Information =====")
        first_row = df.iloc[0]
        print(f"Image name: {first_row['image']}")
        print(f"Game type: {first_row['game']}")
        
        # Print camera information
        print("\n===== Camera Information =====")
        pprint(first_row['camera'])
        
        # Print setup information
        print("\n===== Setup Information =====")
        pprint(first_row['setup'])
        
        # Print card information
        print("\n===== Card Information =====")
        print(f"Number of cards: {first_row['card_number']}")
        print("Card types:")
        pprint(first_row['card_types'])
        
        # Print information about players
        print("\n===== Player Information =====")
        print(f"Number of players: {first_row['player_number']}")
        for i, player in enumerate(first_row['players_info']):
            print(f"\nPlayer {i+1}:")
            print(f"  ID: {player['player_id']}")
            print(f"  Cards: {player['cards']}")
            print(f"  Number of cards: {player['n_cards']}")
            print(f"  Number of piles: {len(player['piles'])}")
            if player['piles']:
                print(f"  First pile chips: {player['piles'][0]['n_chips']}")
        
        # Print community information
        print("\n===== Community Information =====")
        pprint(first_row['community_info'])
        
        # Try getting the cards dataframe
        print("\n===== Cards DataFrame =====")
        try:
            cards_df = poker_dataset.get_cards_dataframe()
            print(f"Cards DataFrame shape: {cards_df.shape}")
            if not cards_df.empty:
                print("First few cards:")
                print(cards_df.head(3))
            else:
                print("No cards found in the cards dataframe.")
        except Exception as e:
            print(f"Error getting cards dataframe: {e}")
        
        # Try getting the players dataframe
        print("\n===== Players DataFrame =====")
        try:
            players_df = poker_dataset.get_players_dataframe()
            print(f"Players DataFrame shape: {players_df.shape}")
            if not players_df.empty:
                print("First few players:")
                print(players_df.head(3))
            else:
                print("No players found in the players dataframe.")
        except Exception as e:
            print(f"Error getting players dataframe: {e}")
        
        # Try getting the card overlap layouts dataframe
        print("\n===== Card Overlap Layouts DataFrame =====")
        try:
            layouts_df = poker_dataset.get_card_overlap_layouts_dataframe()
            print(f"Card Overlap Layouts DataFrame shape: {layouts_df.shape}")
            if not layouts_df.empty:
                print("First few layouts:")
                print(layouts_df.head(3))
            else:
                print("No layouts found in the card overlap layouts dataframe.")
        except Exception as e:
            print(f"Error getting card overlap layouts dataframe: {e}")
        
        print("\n===== Test Complete =====")
        
    except Exception as e:
        print("\n===== ERROR =====")
        print(f"Exception: {type(e).__name__}: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

