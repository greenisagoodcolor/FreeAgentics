"""
Module for FreeAgentics Active Inference implementation.
"""

import heapq
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import h3
import numpy as np

"""
Spatial Computing API for FreeAgentics
This module provides a comprehensive API for spatial operations using H3 hexagonal grid,
including neighbor calculation, pathfinding, resource distribution, and observation modeling.
"""
logger = logging.getLogger(__name__)


@dataclass
class SpatialCoordinate:
    """Represents a coordinate in the spatial system"""

    hex_id: str
    lat: float = field(init=False)
    lng: float = field(init=False)

    def __post_init__(self):
        """Calculate lat/lng from hex_id"""
        self.lat, self.lng = h3.cell_to_latlng(self.hex_id)

    @classmethod
    def from_lat_lng(
            cls,
            lat: float,
            lng: float,
            resolution: int = 7) -> "SpatialCoordinate":
        """Create from latitude/longitude"""
        hex_id = h3.latlng_to_cell(lat, lng, resolution)
        return cls(hex_id=hex_id)

    def distance_to(self, other: "SpatialCoordinate") -> float:
        """Calculate great circle distance in km"""
        return h3.great_circle_distance(
            (self.lat, self.lng), (other.lat, other.lng), unit="km")


class ResourceType(Enum):
    """Types of resources available in the spatial system"""

    FOOD = "food"
    WATER = "water"
    MATERIALS = "materials"
    ENERGY = "energy"
    KNOWLEDGE = "knowledge"
    CUSTOM = "custom"


@dataclass
class ResourceDistribution:
    """Defines how resources are distributed spatially"""

    resource_type: ResourceType
    base_amount: float
    variance: float
    distribution_pattern: str  # 'uniform', 'clustered', 'gradient', 'random'
    cluster_centers: Optional[List[str]] = None
    gradient_origin: Optional[str] = None
    gradient_falloff: float = 0.1

    def calculate_amount(
            self,
            hex_id: str,
            distance_from_origin: float = 0) -> float:
        """Calculate resource amount for a specific hex"""
        if self.distribution_pattern == "uniform":
            return np.random.normal(self.base_amount, self.variance)
        elif self.distribution_pattern == "clustered":
            if not self.cluster_centers:
                return 0
            # Find distance to nearest cluster center
            min_distance = float("inf")
            for center in self.cluster_centers:
                distance = h3.grid_distance(hex_id, center)
                min_distance = min(min_distance, distance)
            # Falloff based on distance
            amount = self.base_amount * math.exp(-min_distance * 0.3)
            return max(0, np.random.normal(amount, self.variance))
        elif self.distribution_pattern == "gradient":
            if not self.gradient_origin:
                return self.base_amount
            # Linear falloff from origin
            distance = h3.grid_distance(hex_id, self.gradient_origin)
            amount = self.base_amount * (1 - distance * self.gradient_falloff)
            return max(0, np.random.normal(amount, self.variance))
        else:  # random
            return np.random.uniform(0, self.base_amount)


@dataclass
class ObservationModel:
    """Models visibility and observation capabilities"""

    base_range: int = 2  # Hexagon rings visible
    elevation_bonus: float = 0.001  # Bonus range per meter of elevation
    terrain_modifiers: Dict[str, float] = field(
        default_factory=lambda: {
            "flat": 1.0,
            "hills": 0.8,
            "mountains": 0.5,
            "forest": 0.7,
            "water": 1.0,
        }
    )
    weather_modifier: float = 1.0  # 0-1, affects visibility

    def calculate_visibility_range(
            self,
            observer_elevation: float,
            observer_terrain: str) -> int:
        """Calculate visibility range based on conditions"""
        terrain_mod = self.terrain_modifiers.get(observer_terrain, 1.0)
        elevation_bonus = int(observer_elevation * self.elevation_bonus)
        total_range = self.base_range + elevation_bonus
        total_range = int(total_range * terrain_mod * self.weather_modifier)
        return max(1, total_range)

    def is_visible(
        self,
        observer_hex: str,
        target_hex: str,
        observer_elevation: float,
        target_elevation: float,
        terrain_heights: Dict[str, float],
    ) -> bool:
        """Check if target is visible from observer considering line of sight"""
        # Get visibility range
        visibility_range = self.calculate_visibility_range(
            observer_elevation, "flat")
        # Check if within range
        distance = h3.grid_distance(observer_hex, target_hex)
        if distance > visibility_range:
            return False
        # Check line of sight
        line = h3.grid_path_cells(observer_hex, target_hex)
        for i, hex_id in enumerate(line[1:-1], 1):  # Skip start and end
            if hex_id not in terrain_heights:
                continue
            # Simple line of sight: check if terrain blocks view
            terrain_height = terrain_heights[hex_id]
            # Calculate expected height at this distance
            t = i / len(line)
            expected_height = observer_elevation * \
                (1 - t) + target_elevation * t
            if terrain_height > expected_height + 10:  # 10m tolerance
                return False
        return True


class SpatialAPI:
    """
    Main API for spatial operations in FreeAgentics.
    Provides high-level functions for:
    - Hexagonal grid operations
    - Neighbor calculations
    - Pathfinding
    - Resource distribution
    - Visibility and observation modeling
    - Spatial queries and caching
    """

    def __init__(self, resolution: int = 7) -> None:
        """
        Initialize the Spatial API.
        Args:
            resolution: H3 resolution level (0-15, default 7 ~5km hexagons)
        """
        self.resolution = resolution
        self._neighbor_cache: Dict[str, List[str]] = {}
        self._distance_cache: Dict[tuple[str, str], int] = {}
        self._path_cache: Dict[tuple[str, str], List[str]] = {}
        self.cache_size_limit = 10000
        # Resource distributions
        self.resource_distributions: Dict[str, ResourceDistribution] = {}
        # Observation model
        self.observation_model = ObservationModel()
        logger.info(f"Initialized SpatialAPI with resolution {resolution}")

    # === Basic H3 Operations ===
    def get_hex_at_position(self, lat: float, lng: float) -> str:
        """Get hex ID at given latitude/longitude"""
        return h3.latlng_to_cell(lat, lng, self.resolution)

    def get_hex_center(self, hex_id: str) -> tuple[float, float]:
        """Get center coordinates of a hex"""
        return h3.cell_to_latlng(hex_id)

    def get_hex_boundary(self, hex_id: str) -> List[tuple[float, float]]:
        """Get boundary coordinates of a hex"""
        return h3.cell_to_boundary(hex_id)

    def get_hex_area(self, hex_id: str) -> float:
        """Get area of a hex in km²"""
        return h3.cell_area(hex_id, unit="km^2")

    # === Neighbor Operations ===
    def get_neighbors(self, hex_id: str, use_cache: bool = True) -> List[str]:
        """Get immediate neighbors of a hex"""
        if use_cache and hex_id in self._neighbor_cache:
            return self._neighbor_cache[hex_id]
        neighbors = list(h3.grid_disk(hex_id, 1))
        neighbors.remove(hex_id)  # Remove center
        if use_cache and len(self._neighbor_cache) < self.cache_size_limit:
            self._neighbor_cache[hex_id] = neighbors
        return neighbors

    def get_neighbors_at_distance(
            self, hex_id: str, k: int) -> Dict[int, List[str]]:
        """Get neighbors grouped by distance (1 to k rings)"""
        rings = h3.grid_ring(hex_id, k)
        return {i: list(ring) for i, ring in enumerate(rings) if i > 0}

    def get_hexes_in_radius(self, hex_id: str, radius_km: float) -> set[str]:
        """Get all hexes within a radius in kilometers"""
        center = SpatialCoordinate(hex_id)
        # Rough estimate, adjust based on resolution
        max_k = int(radius_km / 5) + 2
        hexes_in_radius = set()
        for k in range(max_k):
            ring = h3.grid_disk(hex_id, k)
            for hex in ring:
                coord = SpatialCoordinate(hex)
                if center.distance_to(coord) <= radius_km:
                    hexes_in_radius.add(hex)
        return hexes_in_radius

    # === Distance Calculations ===
    def hex_distance(
            self,
            hex1: str,
            hex2: str,
            use_cache: bool = True) -> int:
        """Get grid distance between two hexes"""
        cache_key = (hex1, hex2) if hex1 < hex2 else (hex2, hex1)
        if use_cache and cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        distance = h3.grid_distance(hex1, hex2)
        if use_cache and len(self._distance_cache) < self.cache_size_limit:
            self._distance_cache[cache_key] = distance
        return distance

    def geodesic_distance(self, hex1: str, hex2: str) -> float:
        """Get geodesic distance in kilometers"""
        coord1 = SpatialCoordinate(hex1)
        coord2 = SpatialCoordinate(hex2)
        return coord1.distance_to(coord2)

    # === Pathfinding ===
    def find_path_simple(self, start: str, goal: str) -> List[str]:
        """Find simple direct path between hexes"""
        return h3.grid_path_cells(start, goal)

    def find_path_astar(
        self,
        start: str,
        goal: str,
        movement_costs: Dict[str, float],
        obstacles: Optional[set[str]] = None,
    ) -> Optional[list[str]]:
        """
        Find optimal path using A* algorithm.
        Args:
            start: Starting hex ID
            goal: Goal hex ID
            movement_costs: Dict mapping hex IDs to movement costs
            obstacles: Set of impassable hex IDs
        Returns:
            List of hex IDs forming the path, or None if no path exists
        """
        obstacles = obstacles or set()
        # Check cache
        cache_key = (start, goal)
        if cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            # Validate cached path is still valid
            if all(h not in obstacles for h in cached_path):
                return cached_path
        # A* implementation
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current == goal:
                # Reconstruct path
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                # Cache the path
                if len(self._path_cache) < self.cache_size_limit:
                    self._path_cache[cache_key] = path
                return path
            for neighbor in self.get_neighbors(current):
                if neighbor in obstacles:
                    continue
                # Calculate new cost
                move_cost = movement_costs.get(neighbor, 1.0)
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.hex_distance(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        return None  # No path found

    # === Resource Distribution ===
    def add_resource_distribution(
            self,
            name: str,
            distribution: ResourceDistribution) -> None:
        """Add a resource distribution pattern"""

        self.resource_distributions[name] = distribution

    def generate_resources(
        self, hexes: List[str], distributions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate resources for given hexes.
        Returns:
            Dict mapping hex IDs to resource amounts
        """
        distributions = distributions or list(
            self.resource_distributions.keys())
        resources = defaultdict(dict)
        for hex_id in hexes:
            for dist_name in distributions:
                if dist_name not in self.resource_distributions:
                    continue
                dist = self.resource_distributions[dist_name]
                amount = dist.calculate_amount(hex_id)
                if amount > 0:
                    resources[hex_id][dist.resource_type.value] = amount
        return dict(resources)

    def find_nearest_resource(
        self,
        start: str,
        resource_type: ResourceType,
        available_resources: Dict[str, Dict[str, float]],
        max_distance: int = 10,
    ) -> Optional[str]:
        """Find nearest hex with specified resource"""
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, distance = queue.popleft()
            if distance > max_distance:
                break
            # Check if current has resource
            if current in available_resources:
                if resource_type.value in available_resources[current]:
                    if available_resources[current][resource_type.value] > 0:
                        return current
            # Add neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    # === Visibility and Observation ===
    def set_observation_model(self, model: ObservationModel) -> None:
        """Set the observation model"""
        self.observation_model = model

    def get_visible_hexes(
        self,
        observer_hex: str,
        observer_elevation: float = 0,
        terrain_heights: Optional[Dict[str, float]] = None,
    ) -> set[str]:
        """Get all hexes visible from observer position"""
        terrain_heights = terrain_heights or {}
        visible = set()
        # Get visibility range
        range_rings = self.observation_model.calculate_visibility_range(
            observer_elevation,
            "flat",  # TODO: pass actual terrain
        )
        # Check each hex in range
        for k in range(1, range_rings + 1):
            for hex_id in h3.grid_disk(observer_hex, k):
                target_elevation = terrain_heights.get(hex_id, 0)
                if self.observation_model.is_visible(
                    observer_hex,
                    hex_id,
                    observer_elevation,
                    target_elevation,
                    terrain_heights,
                ):
                    visible.add(hex_id)
        return visible

    # === Spatial Queries ===
    def query_region(
        self,
        center: str,
        radius_hexes: int,
        filter_func: Optional[Callable[[str], bool]] = None,
    ) -> List[str]:
        """Query hexes in a region with optional filtering"""
        hexes = []
        for hex_id in h3.grid_disk(center, radius_hexes):
            if filter_func is None or filter_func(hex_id):
                hexes.append(hex_id)
        return hexes

    def find_clusters(self,
                      points: List[str],
                      min_cluster_size: int = 3,
                      max_distance: int = 2) -> List[List[str]]:
        """Find spatial clusters of hexes"""
        clusters = []
        unvisited = set(points)
        while unvisited:
            # Start new cluster
            start = unvisited.pop()
            cluster = [start]
            to_check = [start]
            while to_check:
                current = to_check.pop()
                # Check neighbors
                for neighbor in self.get_neighbors(current):
                    if neighbor in unvisited and self.hex_distance(
                            start, neighbor) <= max_distance:
                        unvisited.remove(neighbor)
                        cluster.append(neighbor)
                        to_check.append(neighbor)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        return clusters

    # === Optimization and Caching ===
    def clear_cache(self) -> None:
        """Clear all caches"""
        self._neighbor_cache.clear()
        self._distance_cache.clear()
        self._path_cache.clear()
        logger.info("Cleared spatial caches")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "neighbor_cache": len(self._neighbor_cache),
            "distance_cache": len(self._distance_cache),
            "path_cache": len(self._path_cache),
        }


# Example usage and tests
if __name__ == "__main__":
    # Create API instance
    api = SpatialAPI(resolution=7)
    # Test basic operations
    sf_hex = api.get_hex_at_position(37.7749, -122.4194)
    print(f"San Francisco hex: {sf_hex}")
    # Get neighbors
    neighbors = api.get_neighbors(sf_hex)
    print(f"Neighbors: {len(neighbors)}")
    # Test pathfinding
    target = neighbors[3]
    path = api.find_path_simple(sf_hex, target)
    print(f"Simple path length: {len(path)}")
    # Test resource distribution
    food_dist = ResourceDistribution(
        resource_type=ResourceType.FOOD,
        base_amount=50,
        variance=10,
        distribution_pattern="clustered",
        cluster_centers=[sf_hex],
    )
    api.add_resource_distribution("food", food_dist)
    # Generate resources
    test_hexes = api.get_neighbors_at_distance(sf_hex, 3)
    all_hexes = [h for hexes in test_hexes.values() for h in hexes]
    resources = api.generate_resources(all_hexes[:20])
    print(f"Generated resources for {len(resources)} hexes")
    # Test visibility
    visible = api.get_visible_hexes(sf_hex, observer_elevation=100)
    print(f"Visible hexes from 100m elevation: {len(visible)}")
