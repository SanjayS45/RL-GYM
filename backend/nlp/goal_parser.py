"""
Natural Language Goal Parser

Converts natural language goal descriptions into structured goal specifications.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re
import numpy as np


@dataclass
class ParsedGoal:
    """Structured representation of a parsed goal."""
    
    goal_type: str  # "position", "collect", "avoid", "survival", "custom"
    target: Optional[Any] = None  # Position, items, etc.
    conditions: Dict[str, Any] = field(default_factory=dict)
    reward_config: Dict[str, float] = field(default_factory=dict)
    termination_config: Dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal_type": self.goal_type,
            "target": self.target if not isinstance(self.target, np.ndarray) else self.target.tolist(),
            "conditions": self.conditions,
            "reward_config": self.reward_config,
            "termination_config": self.termination_config,
            "original_text": self.original_text,
            "confidence": self.confidence,
        }


class GoalParser:
    """
    Parse natural language goals into structured specifications.
    
    Uses pattern matching and keyword extraction for goal interpretation.
    Can be extended with ML-based parsing for more complex goals.
    """
    
    # Position keywords and their relative locations
    POSITION_KEYWORDS = {
        "top": (0.5, 0.1),
        "bottom": (0.5, 0.9),
        "left": (0.1, 0.5),
        "right": (0.9, 0.5),
        "center": (0.5, 0.5),
        "middle": (0.5, 0.5),
        "top-left": (0.1, 0.1),
        "top-right": (0.9, 0.1),
        "bottom-left": (0.1, 0.9),
        "bottom-right": (0.9, 0.9),
        "corner": (0.9, 0.9),  # Default to bottom-right
    }
    
    # Action keywords
    ACTION_KEYWORDS = {
        "reach": "position",
        "go": "position",
        "navigate": "position",
        "move": "position",
        "get": "position",
        "collect": "collect",
        "gather": "collect",
        "pick": "collect",
        "avoid": "avoid",
        "dodge": "avoid",
        "evade": "avoid",
        "survive": "survival",
        "stay": "survival",
        "last": "survival",
        "climb": "position",
        "jump": "position",
        "cross": "position",
    }
    
    # Obstacle/challenge keywords
    OBSTACLE_KEYWORDS = [
        "obstacle", "wall", "barrier", "block",
        "gap", "pit", "hole",
        "enemy", "hazard", "danger",
        "platform", "ledge",
    ]
    
    # Time/duration patterns
    TIME_PATTERNS = [
        r"(\d+)\s*(?:second|sec|s)",
        r"(\d+)\s*(?:step|steps)",
        r"(\d+)\s*(?:minute|min|m)",
    ]
    
    def __init__(self, env_width: int = 800, env_height: int = 600):
        """
        Initialize the goal parser.
        
        Args:
            env_width: Environment width for position calculations
            env_height: Environment height for position calculations
        """
        self.env_width = env_width
        self.env_height = env_height
    
    def parse(self, goal_text: str) -> ParsedGoal:
        """
        Parse a natural language goal description.
        
        Args:
            goal_text: Natural language goal description
            
        Returns:
            ParsedGoal: Structured goal specification
        """
        # Normalize text
        text = goal_text.lower().strip()
        
        # Determine goal type
        goal_type, confidence = self._detect_goal_type(text)
        
        # Parse based on type
        if goal_type == "position":
            return self._parse_position_goal(text, confidence)
        elif goal_type == "collect":
            return self._parse_collect_goal(text, confidence)
        elif goal_type == "avoid":
            return self._parse_avoid_goal(text, confidence)
        elif goal_type == "survival":
            return self._parse_survival_goal(text, confidence)
        else:
            return self._parse_generic_goal(text, confidence)
    
    def _detect_goal_type(self, text: str) -> Tuple[str, float]:
        """Detect the type of goal from text."""
        words = text.split()
        
        for word in words:
            if word in self.ACTION_KEYWORDS:
                return self.ACTION_KEYWORDS[word], 0.8
        
        # Check for implicit goal types
        if any(kw in text for kw in ["to the", "toward", "towards"]):
            return "position", 0.6
        if "collect" in text or "item" in text:
            return "collect", 0.7
        if "survive" in text or "alive" in text or "don't die" in text:
            return "survival", 0.7
        
        return "position", 0.4  # Default to position goal
    
    def _parse_position_goal(self, text: str, confidence: float) -> ParsedGoal:
        """Parse a position-based goal."""
        # Extract target position
        position = self._extract_position(text)
        
        # Check for obstacles to navigate
        has_obstacles = any(kw in text for kw in self.OBSTACLE_KEYWORDS)
        
        # Determine reward shaping
        if "quickly" in text or "fast" in text:
            reward_config = {
                "goal_reward": 100.0,
                "time_penalty": 0.1,
                "distance_shaping": 0.2,
            }
        else:
            reward_config = {
                "goal_reward": 100.0,
                "time_penalty": 0.01,
                "distance_shaping": 0.1,
            }
        
        # Termination conditions
        termination_config = {
            "on_goal_reached": True,
            "on_timeout": True,
        }
        
        return ParsedGoal(
            goal_type="position",
            target=position,
            conditions={"navigate_obstacles": has_obstacles},
            reward_config=reward_config,
            termination_config=termination_config,
            original_text=text,
            confidence=confidence,
        )
    
    def _extract_position(self, text: str) -> np.ndarray:
        """Extract target position from text."""
        # Check for explicit coordinates
        coord_match = re.search(r"\((\d+),\s*(\d+)\)", text)
        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
            return np.array([x, y], dtype=np.float32)
        
        # Check for position keywords
        for keyword, (rel_x, rel_y) in self.POSITION_KEYWORDS.items():
            if keyword in text:
                return np.array([
                    rel_x * self.env_width,
                    rel_y * self.env_height
                ], dtype=np.float32)
        
        # Check for directional modifiers
        x_pos = 0.5
        y_pos = 0.5
        
        if "far" in text:
            if "right" in text:
                x_pos = 0.95
            elif "left" in text:
                x_pos = 0.05
            if "up" in text or "top" in text:
                y_pos = 0.05
            elif "down" in text or "bottom" in text:
                y_pos = 0.95
        else:
            if "right" in text:
                x_pos = 0.75
            elif "left" in text:
                x_pos = 0.25
            if "up" in text or "top" in text:
                y_pos = 0.25
            elif "down" in text or "bottom" in text:
                y_pos = 0.75
        
        return np.array([
            x_pos * self.env_width,
            y_pos * self.env_height
        ], dtype=np.float32)
    
    def _parse_collect_goal(self, text: str, confidence: float) -> ParsedGoal:
        """Parse a collection-based goal."""
        # Extract number of items
        num_match = re.search(r"(\d+)\s*(?:item|object|coin|gem)", text)
        num_items = int(num_match.group(1)) if num_match else 3
        
        # Generate random item positions
        items = []
        for _ in range(num_items):
            items.append([
                np.random.uniform(0.1, 0.9) * self.env_width,
                np.random.uniform(0.1, 0.9) * self.env_height
            ])
        
        reward_config = {
            "item_reward": 10.0,
            "completion_bonus": 50.0,
            "time_penalty": 0.01,
        }
        
        return ParsedGoal(
            goal_type="collect",
            target=items,
            conditions={"num_items": num_items},
            reward_config=reward_config,
            termination_config={"on_all_collected": True},
            original_text=text,
            confidence=confidence,
        )
    
    def _parse_avoid_goal(self, text: str, confidence: float) -> ParsedGoal:
        """Parse an avoidance-based goal."""
        # Determine what to avoid
        avoid_type = "obstacles"
        if "enemy" in text or "enemies" in text:
            avoid_type = "enemies"
        elif "hazard" in text:
            avoid_type = "hazards"
        
        reward_config = {
            "survival_reward": 0.1,
            "collision_penalty": -10.0,
        }
        
        return ParsedGoal(
            goal_type="avoid",
            target=avoid_type,
            conditions={"avoid_type": avoid_type},
            reward_config=reward_config,
            termination_config={"on_collision": True},
            original_text=text,
            confidence=confidence,
        )
    
    def _parse_survival_goal(self, text: str, confidence: float) -> ParsedGoal:
        """Parse a survival-based goal."""
        # Extract duration
        duration = 500  # Default steps
        
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                if "second" in pattern or "sec" in pattern:
                    duration = value * 50  # Assuming 50 steps per second
                elif "minute" in pattern or "min" in pattern:
                    duration = value * 3000
                else:
                    duration = value
                break
        
        reward_config = {
            "survival_reward_per_step": 0.1,
            "completion_bonus": 100.0,
        }
        
        return ParsedGoal(
            goal_type="survival",
            target=duration,
            conditions={"duration_steps": duration},
            reward_config=reward_config,
            termination_config={"on_duration": True},
            original_text=text,
            confidence=confidence,
        )
    
    def _parse_generic_goal(self, text: str, confidence: float) -> ParsedGoal:
        """Parse a generic/unclear goal."""
        # Default to position goal with center-right target
        return ParsedGoal(
            goal_type="position",
            target=np.array([self.env_width * 0.8, self.env_height * 0.5]),
            conditions={},
            reward_config={
                "goal_reward": 100.0,
                "distance_shaping": 0.1,
            },
            termination_config={"on_goal_reached": True},
            original_text=text,
            confidence=confidence * 0.5,  # Lower confidence for generic
        )
    
    def explain(self, parsed_goal: ParsedGoal) -> str:
        """
        Generate a human-readable explanation of the parsed goal.
        
        Args:
            parsed_goal: The parsed goal to explain
            
        Returns:
            Human-readable explanation
        """
        lines = [f"Goal Type: {parsed_goal.goal_type.capitalize()}"]
        lines.append(f"Confidence: {parsed_goal.confidence:.0%}")
        
        if parsed_goal.goal_type == "position":
            target = parsed_goal.target
            lines.append(f"Target Position: ({target[0]:.0f}, {target[1]:.0f})")
            if parsed_goal.conditions.get("navigate_obstacles"):
                lines.append("Must navigate around obstacles")
        
        elif parsed_goal.goal_type == "collect":
            lines.append(f"Items to collect: {parsed_goal.conditions.get('num_items', 'Unknown')}")
            lines.append(f"Reward per item: {parsed_goal.reward_config.get('item_reward', 10)}")
        
        elif parsed_goal.goal_type == "survival":
            duration = parsed_goal.conditions.get("duration_steps", 500)
            lines.append(f"Survive for: {duration} steps")
        
        lines.append("\nReward Configuration:")
        for key, value in parsed_goal.reward_config.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


# Example usage and testing
EXAMPLE_GOALS = [
    "Reach the top-right corner",
    "Navigate to the bottom while avoiding obstacles",
    "Collect 5 items scattered across the map",
    "Survive for 100 steps",
    "Go to the far right as quickly as possible",
    "Climb over the obstacle and reach the platform",
    "Avoid all enemies and get to the exit",
]

