"""
Module for FreeAgentics Active Inference implementation.
"""

import time

import pytest

from world.h3_world import H3World
from world.spatial import Biome, ObservationModel, ResourceDistribution, ResourceType, TerrainType
from world.spatial.spatial_api import SpatialAPI
from typing import Optional


class TestSpatialIntegration:
    """Test suite for spatial module integration."""

    @pytest.fixture
    def spatial_api(self):
        """Create a SpatialAPI instance."""
        return SpatialAPI(resolution=7)

    @pytest.fixture
    def h3_world(self):
        """Create an H3World instance."""
        return H3World(center_lat=37.7749, center_lng=-122.4194, resolution=7, num_rings=5, seed=42)

    def test_api_world_coordinate_consistency(self, spatial_api, h3_world) -> None:
        """Test that coordinates are consistent between API and world."""
        center_hex = h3_world.center_hex
        world_coords = h3_world.get_cell(center_hex).coordinates
        api_coords = spatial_api.get_hex_center(center_hex)
        assert world_coords == api_coords

    def test_neighbor_calculations_match(self, spatial_api, h3_world) -> None:
        """Test that neighbor calculations are consistent."""
        center_hex = h3_world.center_hex
        world_neighbors = h3_world.get_neighbors(center_hex)
        world_neighbor_ids = {n.hex_id for n in world_neighbors}
        api_neighbor_ids = set(spatial_api.get_neighbors(center_hex))
        assert world_neighbor_ids == api_neighbor_ids

    def test_pathfinding_respects_terrain(self, spatial_api, h3_world) -> None:
        """Test that pathfinding considers terrain movement costs."""
        cells = list(h3_world.cells.keys())
        start_hex = cells[0]
        end_hex = cells[10]
        movement_costs = (
            {hex_id: cell.movement_cost for hex_id, cell in h3_world.cells.items()})
        obstacles = {
            hex_id for hex_id, cell in h3_world.cells.items() if cell.terrain == TerrainType.WATER
        }
        path = spatial_api.find_path_astar(start_hex, end_hex, movement_costs, obstacles)
        if path:
            for hex_id in path:
                assert hex_id not in obstacles
            total_cost = sum(movement_costs.get(h, 1.0) for h in path[1:])
            assert total_cost > 0

    def test_resource_distribution_in_world(self, spatial_api, h3_world) -> None:
        """Test resource distribution across world hexes."""
        forest_centers = [
            hex_id for hex_id, cell in h3_world.cells.items() if cell.biome == Biome.FOREST
        ][:3]
        food_dist = ResourceDistribution(
            resource_type=ResourceType.FOOD,
            base_amount=50.0,
            variance=10.0,
            distribution_pattern="clustered",
            cluster_centers=forest_centers,
        )
        spatial_api.add_resource_distribution("forest_food", food_dist)
        all_hexes = list(h3_world.cells.keys())
        resources = spatial_api.generate_resources(all_hexes, ["forest_food"])
        for center in forest_centers:
            if center in resources and "food" in resources[center]:
                assert resources[center]["food"] > 0

    def test_visibility_with_elevation(self, spatial_api, h3_world) -> None:
        """Test visibility calculations considering terrain elevation."""
        mountain_hex = None
        for hex_id, cell in h3_world.cells.items():
            if cell.terrain == TerrainType.MOUNTAINS:
                mountain_hex = hex_id
                break
        if mountain_hex:
            mountain_cell = h3_world.get_cell(mountain_hex)
            terrain_heights = {hex_id: cell.elevation for hex_id, cell in h3_world.cells.items()}
            obs_model = ObservationModel(base_range=2, elevation_bonus=0.002)
            spatial_api.set_observation_model(obs_model)
            visible = spatial_api.get_visible_hexes(
                mountain_hex, mountain_cell.elevation, terrain_heights
            )
            assert len(visible) > 0
            flat_hex = None
            for hex_id, cell in h3_world.cells.items():
                if cell.terrain == TerrainType.FLAT:
                    flat_hex = hex_id
                    break
            if flat_hex:
                flat_cell = h3_world.get_cell(flat_hex)
                flat_visible = spatial_api.get_visible_hexes(
                    flat_hex, flat_cell.elevation, terrain_heights
                )
                if mountain_cell.elevation > flat_cell.elevation + 500:
                    assert len(visible) >= len(flat_visible)

    def test_find_nearest_resource_in_world(self, spatial_api, h3_world) -> None:
        """Test finding nearest resource using world data."""
        world_resources = {}
        for hex_id, cell in h3_world.cells.items():
            if any(cell.resources.values()):
                world_resources[hex_id] = cell.resources
        start_hex = None
        for hex_id, cell in h3_world.cells.items():
            if cell.biome == Biome.DESERT:
                start_hex = hex_id
                break
        if start_hex:
            nearest_water = spatial_api.find_nearest_resource(
                start_hex, ResourceType.WATER, world_resources, max_distance=20
            )
            if nearest_water:
                assert world_resources[nearest_water]["water"] > 0
                water_cell = h3_world.get_cell(nearest_water)
                assert water_cell.biome != Biome.DESERT or water_cell.resources["water"] > 0

    def test_spatial_queries_on_world(self, spatial_api, h3_world) -> None:
        """Test spatial queries with world data."""
        center_hex = h3_world.center_hex

        def is_forest(hex_id):
            cell = h3_world.get_cell(hex_id)
            return cell and cell.biome == Biome.FOREST

        forest_hexes = spatial_api.query_region(center_hex, radius_hexes=3, filter_func=is_forest)
        for hex_id in forest_hexes:
            cell = h3_world.get_cell(hex_id)
            assert cell.biome == Biome.FOREST

    def test_agent_movement_path_cost(self, spatial_api, h3_world) -> None:
        """Test calculating movement costs for agent paths."""
        cells = list(h3_world.cells.keys())
        start = cells[0]
        end = cells[20]
        simple_path = spatial_api.find_path_simple(start, end)
        simple_cost = h3_world.get_total_path_cost(simple_path)
        movement_costs = {hex_id: cell.movement_cost for hex_id, cell in h3_world.cells.items()}
        optimal_path = spatial_api.find_path_astar(start, end, movement_costs)
        if optimal_path:
            optimal_cost = h3_world.get_total_path_cost(optimal_path)
            assert optimal_cost <= simple_cost + 0.1

    def test_cache_performance(self, spatial_api, h3_world) -> None:
        """Test that caching improves performance."""
        spatial_api.clear_cache()
        hexes = list(h3_world.cells.keys())[:50]
        start_time = time.time()
        for hex_id in hexes:
            _ = spatial_api.get_neighbors(hex_id, use_cache=False)
        no_cache_time = time.time() - start_time
        start_time = time.time()
        for hex_id in hexes:
            _ = spatial_api.get_neighbors(hex_id, use_cache=True)
        first_cache_time = time.time() - start_time
        start_time = time.time()
        for hex_id in hexes:
            _ = spatial_api.get_neighbors(hex_id, use_cache=True)
        cached_time = time.time() - start_time
        assert cached_time < no_cache_time * 0.5
        stats = spatial_api.get_cache_stats()
        assert stats["neighbor_cache"] == len(hexes)

    def test_resource_clusters_in_biomes(self, spatial_api, h3_world) -> None:
        """Test finding resource clusters in specific biomes."""
        food_hexes = []
        for hex_id, cell in h3_world.cells.items():
            if cell.biome in [Biome.FOREST, Biome.JUNGLE]:
                if cell.resources.get("food", 0) > 30:
                    food_hexes.append(hex_id)
        if len(food_hexes) > 5:
            clusters = spatial_api.find_clusters(food_hexes, min_cluster_size=2, max_distance=3)
            assert len(clusters) > 0
            for cluster in clusters:
                assert len(cluster) >= 2
                assert all(h in food_hexes for h in cluster)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
