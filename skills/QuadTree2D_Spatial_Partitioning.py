import pathlib
import json
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum, auto


class QuadTreeOperation(Enum):
    INSERT = auto()
    QUERY_RANGE = auto()
    CHECK_COLLISION = auto()


@dataclass
class Point:
    x: float
    y: float
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    def contains(self, point: Point) -> bool:
        return (self.x_min <= point.x <= self.x_max and 
                self.y_min <= point.y <= self.y_max)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        return (not (self.x_max < other.x_min or 
                    self.x_min > other.x_max or 
                    self.y_max < other.y_min or 
                    self.y_min > other.y_max))


class QuadTreeNode:
    def __init__(self, boundary: BoundingBox, capacity: int = 4):
        self.boundary = boundary
        self.capacity = capacity
        self.points: List[Point] = []
        self.subdivided = False
        self.subdivisions: List['QuadTreeNode'] = []

    def subdivide(self) -> None:
        x_mid = (self.boundary.x_min + self.boundary.x_max) / 2
        y_mid = (self.boundary.y_min + self.boundary.y_max) / 2
        
        self.subdivisions = [
            QuadTreeNode(BoundingBox(self.boundary.x_min, y_mid, x_mid, self.boundary.y_max), self.capacity),
            QuadTreeNode(BoundingBox(x_mid, y_mid, self.boundary.x_max, self.boundary.y_max), self.capacity),
            QuadTreeNode(BoundingBox(self.boundary.x_min, self.boundary.y_min, x_mid, y_mid), self.capacity),
            QuadTreeNode(BoundingBox(x_mid, self.boundary.y_min, self.boundary.x_max, y_mid), self.capacity)
        ]
        self.subdivided = True

    def insert(self, point: Point) -> bool:
        if not self.boundary.contains(point):
            return False
            
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
            
        if not self.subdivided:
            self.subdivide()
            
        for subdivision in self.subdivisions:
            if subdivision.insert(point):
                return True
                
        return False

    def query_range(self, query_box: BoundingBox) -> List[Point]:
        points_in_range = []
        
        if not self.boundary.intersects(query_box):
            return points_in_range
            
        for point in self.points:
            if query_box.contains(point):
                points_in_range.append(point)
                
        if self.subdivided:
            for subdivision in self.subdivisions:
                points_in_range.extend(subdivision.query_range(query_box))
                
        return points_in_range

    def check_collision(self, point: Point) -> List[Point]:
        collisions = []
        
        if not self.boundary.contains(point):
            return collisions
            
        for p in self.points:
            if p.x == point.x and p.y == point.y:
                collisions.append(p)
                
        if self.subdivided:
            for subdivision in self.subdivisions:
                collisions.extend(subdivision.check_collision(point))
                
        return collisions


class QuadTree2D:
    def __init__(self, boundary: BoundingBox, capacity: int = 4):
        self.root = QuadTreeNode(boundary, capacity)

    def insert_point(self, point: Point) -> bool:
        return self.root.insert(point)

    def query_range(self, query_box: BoundingBox) -> List[Point]:
        return self.root.query_range(query_box)

    def check_collision(self, point: Point) -> List[Point]:
        return self.root.check_collision(point)


def execute(**kwargs) -> dict:
    result = {"result": {}}
    
    try:
        operation = kwargs.get("operation", QuadTreeOperation.INSERT.value)
        
        if operation == QuadTreeOperation.INSERT.value:
            x = float(kwargs.get("x", 0))
            y = float(kwargs.get("y", 0))
            boundary = BoundingBox(-1000, -1000, 1000, 1000)
            quadtree = QuadTree2D(boundary)
            point = Point(x, y)
            success = quadtree.insert_point(point)
            result["result"] = {
                "operation": "insert",
                "point": {"x": x, "y": y},
                "success": success
            }
            
        elif operation == QuadTreeOperation.QUERY_RANGE.value:
            x_min = float(kwargs.get("x_min", -1000))
            y_min = float(kwargs.get("y_min", -1000))
            x_max = float(kwargs.get("x_max", 1000))
            y_max = float(kwargs.get("y_max", 1000))
            boundary = BoundingBox(-1000, -1000, 1000, 1000)
            quadtree = QuadTree2D(boundary)
            
            # Insert some points for testing
            test_points = [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
                Point(3, 3),
                Point(5, 5),
                Point(10, 10)
            ]
            for point in test_points:
                quadtree.insert_point(point)
                
            query_box = BoundingBox(x_min, y_min, x_max, y_max)
            points_in_range = quadtree.query_range(query_box)
            result["result"] = {
                "operation": "query_range",
                "query_box": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                },
                "points_in_range": [{"x": p.x, "y": p.y} for p in points_in_range]
            }
            
        elif operation == QuadTreeOperation.CHECK_COLLISION.value:
            x = float(kwargs.get("x", 0))
            y = float(kwargs.get("y", 0))
            boundary = BoundingBox(-1000, -1000, 1000, 1000)
            quadtree = QuadTree2D(boundary)
            
            # Insert some points for testing
            test_points = [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
                Point(3, 3),
                Point(5, 5),
                Point(10, 10)
            ]
            for point in test_points:
                quadtree.insert_point(point)
                
            point = Point(x, y)
            collisions = quadtree.check_collision(point)
            result["result"] = {
                "operation": "check_collision",
                "point": {"x": x, "y": y},
                "collisions": [{"x": p.x, "y": p.y} for p in collisions]
            }
            
        else:
            result["result"] = {"error": "Invalid operation"}
            
    except Exception as e:
        result["result"] = {"error": str(e)}
        
    return result