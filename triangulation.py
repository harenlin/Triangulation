import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

class Triangle:
    def __init__(self, points):
        self.points = points
        self.circumcenter, self.circumradius = self.circumcircle()

    def circumcircle(self):
        A = np.array([
            [self.points[0][0], self.points[0][1], 1],
            [self.points[1][0], self.points[1][1], 1],
            [self.points[2][0], self.points[2][1], 1]
        ])
        D = 2 * np.linalg.det(A)
        
        Ux = (self.points[0][0]**2 + self.points[0][1]**2) * (self.points[1][1] - self.points[2][1]) + \
             (self.points[1][0]**2 + self.points[1][1]**2) * (self.points[2][1] - self.points[0][1]) + \
             (self.points[2][0]**2 + self.points[2][1]**2) * (self.points[0][1] - self.points[1][1])
        
        Uy = (self.points[0][0]**2 + self.points[0][1]**2) * (self.points[2][0] - self.points[1][0]) + \
             (self.points[1][0]**2 + self.points[1][1]**2) * (self.points[0][0] - self.points[2][0]) + \
             (self.points[2][0]**2 + self.points[2][1]**2) * (self.points[1][0] - self.points[0][0])
        
        circumcenter = np.array([Ux, Uy]) / D
        circumradius = np.linalg.norm(circumcenter - self.points[0])
        
        return circumcenter, circumradius

    def contains_point_in_circumcircle(self, point):
        return np.linalg.norm(point - self.circumcenter) <= self.circumradius

def bowyer_watson(points):
    super_triangle = Triangle(np.array([[10, 10], [-10, 10], [0, -10]]))
    triangles = [super_triangle]

    for point in points:
        bad_triangles = [t for t in triangles if t.contains_point_in_circumcircle(point)]
        polygon = []

        for t in bad_triangles:
            for i in range(3):
                edge = (t.points[i], t.points[(i+1) % 3])
                if not any(np.array_equal(edge[::-1], (bt.points[j], bt.points[(j+1) % 3])) for bt in bad_triangles for j in range(3)):
                    polygon.append(edge)
                    
        triangles = [t for t in triangles if t not in bad_triangles]

        for edge in polygon:
            triangles.append(Triangle(np.array([edge[0], edge[1], point])))

    return [t for t in triangles if all(p not in super_triangle.points for p in t.points)]

def plot_triangulation(triangles, points):
    # Extract triangle vertices
    triangles_indices = []
    for t in triangles:
        indices = [np.where((points == p).all(axis=1))[0][0] for p in t.points]
        triangles_indices.append(indices)
    
    # Convert to numpy arrays
    triangles_indices = np.array(triangles_indices)
    
    # Plot using Triangulation
    plt.triplot(Triangulation(points[:, 0], points[:, 1], triangles_indices), 'bo-')
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.show()

points = np.random.rand(10, 2)
triangles = bowyer_watson(points)
plot_triangulation(triangles, points)

