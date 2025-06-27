import json
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import h3
import numpy as np

"""
H3 Grid World Implementation
Hexagonal world using Uber's H3 system for agent environment.
"""
logger = logging.getLogger(__name__)


class Biome(Enum):
    """Available biome types."""

    ARCTIC = "arctic"
    TUNDRA = "tundra"
    FOREST = "forest"
    GRASSLAND = "grassland"
    DESERT = "desert"
    SAVANNA = "savanna"
    JUNGLE = "jungle"
    MOUNTAIN = "mountain"
    COASTAL = "coastal"
    OCEAN = "ocean"


class TerrainType(Enum):
    """Terrain types affecting movement."""

    FLAT = "flat"
    HILLS = "hills"
    MOUNTAINS = "mountains"
    WATER = "water"
    MARSH = "marsh"
    SAND = "sand"


@dataclass
class HexCell:
    """Single hexagonal cell in the world."""

    hex_id: str
    biome: Biome
    terrain: TerrainType
    elevation: float  # 0-1000 meters
    temperature: float  # Celsius
    moisture: float  # 0-100%
    resources: Dict[str, float]
    visibility_range: int = 2  # How far agents can see from this cell
    movement_cost: float = 1.0  # Energy cost to enter this cell

    @property
    def coordinates(self) -> tuple[float, float]:
        """Get lat/lng coordinates of the cell center."""
        result = h3.cell_to_latlng(self.hex_id)
        return (float(result[0]), float(result[1]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hex_id": self.hex_id,
            "biome": self.biome.value,
            "terrain": self.terrain.value,
            "elevation": self.elevation,
            "temperature": self.temperature,
            "moisture": self.moisture,
            "resources": self.resources,
            "visibility_range": self.visibility_range,
            "movement_cost": self.movement_cost,
            "coordinates": self.coordinates,
        }


class H3World:
    """
    Hexagonal grid world using H3 system.
    Creates a world with concentric rings of hexagons,
    varied biomes, terrain, and resources.
    """

    DEFAULT_RESOLUTION = 7  # H3 resolution (7 = ~5km hexagons)
    DEFAULT_RINGS = 10  # Number of rings from center

    def __init__(
        self,
        center_lat: float = 0.0,
        center_lng: float = 0.0,
        resolution: int = DEFAULT_RESOLUTION,
        num_rings: int = DEFAULT_RINGS,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the H3 world.
        Args:
            center_lat: Latitude of world center
            center_lng: Longitude of world center
            resolution: H3 resolution (0-15, higher = smaller cells)
            num_rings: Number of hexagon rings from center
            seed: Random seed for reproducible worlds
        """
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.resolution = resolution
        self.num_rings = num_rings
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        # Get center hex
        self.center_hex = h3.latlng_to_cell(center_lat, center_lng, resolution)
        # Storage for world cells
        self.cells: Dict[str, HexCell] = {}
        # Generate the world
        self._generate_world()
        logger.info(f"Created H3World with {len(self.cells)} cells at resolution {resolution}")

    def _generate_world(self):
        """Generate the hexagonal world with biomes and terrain."""
        # Get all hexagons within num_rings of center
        hex_ids = []
        for ring in range(self.num_rings + 1):
            if ring == 0:
                hex_ids.append([self.center_hex])
            else:
                hex_ids.append(h3.grid_ring(self.center_hex, ring))
        for ring_num, ring_hexes in enumerate(hex_ids):
            for hex_id in ring_hexes:
                # Generate cell properties
                cell = self._generate_cell(hex_id, ring_num)
                self.cells[hex_id] = cell

    def _generate_cell(self, hex_id: str, ring_distance: int) -> HexCell:
        """Generate a single cell with biome, terrain, and resources."""
        lat, lng = h3.cell_to_latlng(hex_id)
        # Calculate base properties
        # Temperature decreases with latitude and varies with "seasons"
        base_temp = 20 + 10 * math.sin(math.radians(lat))
        temp_variation = 5 * math.sin(lng / 10)
        temperature = base_temp + temp_variation
        # Moisture based on position and some noise
        base_moisture = 50 + 30 * math.cos(math.radians(lng))
        moisture_noise = np.random.normal(0, 10)
        moisture = np.clip(base_moisture + moisture_noise, 0, 100)
        # Elevation based on distance from center and noise
        base_elevation = 100 + ring_distance * 50
        elevation_noise = np.random.normal(0, 100)
        elevation = max(0, base_elevation + elevation_noise)
        # Determine biome based on temperature and moisture
        biome = self._calculate_biome(temperature, moisture, elevation)
        # Determine terrain based on elevation and biome
        terrain = self._calculate_terrain(elevation, biome)
        # Generate resources based on biome
        resources = self._generate_resources(biome, terrain)
        # Calculate movement cost
        movement_cost = self._calculate_movement_cost(terrain, elevation)
        # Visibility range affected by terrain
        visibility_range = self._calculate_visibility(terrain, elevation)
        return HexCell(
            hex_id=hex_id,
            biome=biome,
            terrain=terrain,
            elevation=elevation,
            temperature=temperature,
            moisture=moisture,
            resources=resources,
            visibility_range=visibility_range,
            movement_cost=movement_cost,
        )

    def _calculate_biome(self, temperature: float, moisture: float, elevation: float) -> Biome:
        """Calculate biome based on environmental factors."""
        # High elevation = mountain
        if elevation > 800:
            return Biome.MOUNTAIN
        # Temperature-based primary classification
        if temperature < -10:
            return Biome.ARCTIC
        elif temperature < 0:
            return Biome.TUNDRA
        elif temperature < 10:
            if moisture > 60:
                return Biome.FOREST
            else:
                return Biome.GRASSLAND
        elif temperature < 20:
            if moisture > 70:
                return Biome.JUNGLE
            elif moisture > 40:
                return Biome.FOREST
            elif moisture > 20:
                return Biome.GRASSLAND
            else:
                return Biome.DESERT
        else:  # Hot climates
            if moisture > 60:
                return Biome.JUNGLE
            elif moisture > 30:
                return Biome.SAVANNA
            else:
                return Biome.DESERT

    def _calculate_terrain(self, elevation: float, biome: Biome) -> TerrainType:
        """Calculate terrain type based on elevation and biome."""
        if biome == Biome.OCEAN:
            return TerrainType.WATER
        elif biome == Biome.MOUNTAIN or elevation > 600:
            return TerrainType.MOUNTAINS
        elif elevation > 300:
            return TerrainType.HILLS
        elif biome in [Biome.DESERT, Biome.COASTAL]:
            return TerrainType.SAND
        elif biome == Biome.JUNGLE and np.random.random() < 0.3:
            return TerrainType.MARSH
        else:
            return TerrainType.FLAT

    def _generate_resources(self, biome: Biome, terrain: TerrainType) -> Dict[str, float]:
        """Generate resources based on biome and terrain."""
        resources = {
            "food": 0.0,
            "water": 0.0,
            "materials": 0.0,
            "energy": 0.0,
            "knowledge": 0.0,
        }
        # Biome-specific resource generation
        if biome == Biome.FOREST:
            resources["food"] = np.random.uniform(20, 50)
            resources["water"] = np.random.uniform(30, 60)
            resources["materials"] = np.random.uniform(40, 80)
        elif biome == Biome.JUNGLE:
            resources["food"] = np.random.uniform(40, 80)
            resources["water"] = np.random.uniform(50, 90)
            resources["materials"] = np.random.uniform(30, 60)
            resources["knowledge"] = np.random.uniform(10, 30)  # Rare plants
        elif biome == Biome.GRASSLAND:
            resources["food"] = np.random.uniform(30, 60)
            resources["water"] = np.random.uniform(20, 40)
            resources["energy"] = np.random.uniform(20, 40)  # Wind/solar
        elif biome == Biome.DESERT:
            resources["food"] = np.random.uniform(0, 20)
            resources["water"] = np.random.uniform(0, 10)
            resources["energy"] = np.random.uniform(40, 80)  # Solar
            resources["materials"] = np.random.uniform(20, 40)  # Minerals
        elif biome == Biome.COASTAL:
            resources["food"] = np.random.uniform(40, 70)  # Fish
            resources["water"] = np.random.uniform(60, 90)
            resources["knowledge"] = np.random.uniform(20, 40)  # Marine life
        elif biome == Biome.MOUNTAIN:
            resources["materials"] = np.random.uniform(50, 90)  # Minerals
            resources["knowledge"] = np.random.uniform(30, 50)  # Rare finds
            resources["water"] = np.random.uniform(20, 50)  # Mountain streams
        elif biome == Biome.ARCTIC or biome == Biome.TUNDRA:
            resources["food"] = np.random.uniform(0, 15)
            resources["water"] = np.random.uniform(30, 60)  # Ice
            resources["knowledge"] = np.random.uniform(40, 60)  # Unique ecosystem
        # Add some randomness
        for resource in resources:
            resources[resource] *= np.random.uniform(0.8, 1.2)
            resources[resource] = max(0, min(100, resources[resource]))
        return resources

    def _calculate_movement_cost(self, terrain: TerrainType, elevation: float) -> float:
        """Calculate movement cost based on terrain."""
        base_costs = {
            TerrainType.FLAT: 1.0,
            TerrainType.HILLS: 1.5,
            TerrainType.MOUNTAINS: 3.0,
            TerrainType.WATER: 5.0,  # Need boat
            TerrainType.MARSH: 2.0,
            TerrainType.SAND: 1.8,
        }
        cost = base_costs.get(terrain, 1.0)
        # Additional cost for elevation changes
        elevation_factor = 1.0 + (elevation / 1000) * 0.5
        return cost * elevation_factor

    def _calculate_visibility(self, terrain: TerrainType, elevation: float) -> int:
        """Calculate visibility range from terrain and elevation."""
        base_visibility = {
            TerrainType.FLAT: 3,
            TerrainType.HILLS: 2,
            TerrainType.MOUNTAINS: 1,
            TerrainType.WATER: 3,
            TerrainType.MARSH: 1,
            TerrainType.SAND: 3,
        }
        visibility = base_visibility.get(terrain, 2)
        # Bonus visibility for high elevation
        if elevation > 500:
            visibility += 1
        if elevation > 800:
            visibility += 1
        return visibility

    def get_cell(self, hex_id: str) -> Optional[HexCell]:
        """Get a cell by its hex ID."""
        return self.cells.get(hex_id)

    def get_neighbors(self, hex_id: str) -> List[HexCell]:
        """Get all neighboring cells."""
        neighbor_ids = h3.grid_disk(hex_id, 1)
        neighbor_ids.remove(hex_id)  # Remove center
        neighbors = []
        for nid in neighbor_ids:
            if nid in self.cells:
                neighbors.append(self.cells[nid])
        return neighbors

    def get_cells_in_range(self, hex_id: str, range: int) -> List[HexCell]:
        """Get all cells within a given range."""
        cell_ids = h3.grid_disk(hex_id, range)
        cells = []
        for cid in cell_ids:
            if cid in self.cells:
                cells.append(self.cells[cid])
        return cells

    def get_visible_cells(self, hex_id: str) -> List[HexCell]:
        """Get all cells visible from a given position."""
        cell = self.get_cell(hex_id)
        if not cell:
            return []
        return self.get_cells_in_range(hex_id, cell.visibility_range)

    def calculate_path(self, start_hex: str, end_hex: str) -> Optional[list[str]]:
        """Calculate optimal path between two hexes (A* pathfinding)."""
        if start_hex not in self.cells or end_hex not in self.cells:
            return None
        # Simple h3 line for now - could be enhanced with A*
        path = h3.grid_path_cells(start_hex, end_hex)
        # Filter to only include cells in our world
        valid_path = [hex_id for hex_id in path if hex_id in self.cells]
        return valid_path if valid_path else None

    def get_total_path_cost(self, path: List[str]) -> float:
        """Calculate total movement cost for a path."""
        if not path:
            return 0.0
        total_cost = 0.0
        for hex_id in path[1:]:  # Skip starting position
            cell = self.get_cell(hex_id)
            if cell:
                total_cost += cell.movement_cost
        return total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert world to dictionary for serialization."""
        return {
            "center_lat": self.center_lat,
            "center_lng": self.center_lng,
            "resolution": self.resolution,
            "num_rings": self.num_rings,
            "seed": self.seed,
            "num_cells": len(self.cells),
            "cells": {hex_id: cell.to_dict() for hex_id, cell in self.cells.items()},
        }

    def save_to_file(self, filepath: str) -> None:
        """Save world to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "H3World":
        """Load world from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        # Create world with same parameters
        world = cls(
            center_lat=data["center_lat"],
            center_lng=data["center_lng"],
            resolution=data["resolution"],
            num_rings=data["num_rings"],
            seed=data["seed"],
        )
        # Load cells
        for hex_id, cell_data in data["cells"].items():
            cell = HexCell(
                hex_id=hex_id,
                biome=Biome(cell_data["biome"]),
                terrain=TerrainType(cell_data["terrain"]),
                elevation=cell_data["elevation"],
                temperature=cell_data["temperature"],
                moisture=cell_data["moisture"],
                resources=cell_data["resources"],
                visibility_range=cell_data["visibility_range"],
                movement_cost=cell_data["movement_cost"],
            )
            world.cells[hex_id] = cell
        return world

    def add_resource(self, hex_id: str, resource_type: str, amount: float) -> bool:
        """Add resources to a specific cell."""
        if hex_id not in self.cells:
            return False
        cell = self.cells[hex_id]
        if resource_type not in cell.resources:
            cell.resources[resource_type] = 0
        cell.resources[resource_type] += amount
        return True


# Example usage
if __name__ == "__main__":
    # Create a test world
    world = H3World(
        center_lat=37.7749,  # San Francisco
        center_lng=-122.4194,
        resolution=7,
        num_rings=5,
        seed=42,
    )
    print(f"Created world with {len(world.cells)} cells")
    # Get center cell
    center_cell = world.get_cell(world.center_hex)
    print("\nCenter cell:")
    if center_cell:
        print(f"  Biome: {center_cell.biome.value}")
        print(f"  Terrain: {center_cell.terrain.value}")
        print(f"  Temperature: {center_cell.temperature:.1f}°C")
        print(f"  Resources: {center_cell.resources}")
    # Get neighbors
    neighbors = world.get_neighbors(world.center_hex)
    print(f"\nFound {len(neighbors)} neighbors")
    # Save world
    world.save_to_file("test_world.json")
