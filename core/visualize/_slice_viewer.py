import numpy as np
from scipy.spatial import Delaunay
import pyglet
from pyglet.gl import *

class SliceViewer(pyglet.window.Window):
    def __init__(self, slices: list[list[np.ndarray]], width=1280, height=720):
        super(SliceViewer, self).__init__(width=width, height=height, resizable=True)
        self.slices = slices
        self.rx, self.ry, self.rz = 0, 0, 0  # Rotation angles for X, Y, Z
        self.zoom = -100  # Zoom factor

        # Enable OpenGL states for better rendering
        self.setup_opengl()

    def setup_opengl(self):
        # Enable depth test for correct rendering order of 3D elements
        glEnable(GL_DEPTH_TEST)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable lighting and a single light source
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        # Set the light position
        light_pos = (1.0, 1.0, 1.0, 0.0)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(*light_pos))

        # Set light properties
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.8, 0.8, 0.8, 1.0))  # Diffuse light
        glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(1.0, 1.0, 1.0, 1.0))  # Specular light

        # Set background color
        glClearColor(1, 1, 1, 1)

        # Set material properties for the slices (grey color)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(0.5, 0.5, 0.5, 1.0))  # Grey
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(0.8, 0.8, 0.8, 1.0))  # Specular highlight
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)  # Shininess for specular highlights

    def on_draw(self):
        self.clear()
        glLoadIdentity()

        # Move the camera and apply rotations in X, Y, and Z axes
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rx, 1, 0, 0)  # Rotate around X-axis
        glRotatef(self.ry, 0, 1, 0)  # Rotate around Y-axis
        glRotatef(self.rz, 0, 0, 1)  # Rotate around Z-axis

        # Render each slice with lighting
        for slice_3d_list in self.slices:
            for slice_3d in slice_3d_list:
                self.draw_slice(slice_3d)

    def draw_slice(self, slice_3d):
        """Triangulate the 2D slice and render it as a 3D surface with normals and lighting."""
        # Perform Delaunay triangulation on the 2D projection of the slice (ignoring z-axis)
        points_2d = slice_3d[:, :2]  # Extract only x, y for 2D triangulation
        tri = Delaunay(points_2d)

        # Render the triangles with normals
        glBegin(GL_TRIANGLES)
        for simplex in tri.simplices:
            # Get the vertices of the triangle
            v0, v1, v2 = slice_3d[simplex]

            # Compute the normal of the triangle
            normal = self.compute_normal(v0, v1, v2)
            glNormal3f(*normal)

            # Draw the vertices of the triangle
            glVertex3f(*v0)
            glVertex3f(*v1)
            glVertex3f(*v2)
        glEnd()

    def compute_normal(self, v0, v1, v2):
        """Compute the normal vector for a triangle defined by vertices v0, v1, and v2."""
        # Calculate the vectors from v0 to v1 and v0 to v2
        vec1 = v1 - v0
        vec2 = v2 - v0

        # Compute the cross product of vec1 and vec2 to get the normal
        normal = np.cross(vec1, vec2)

        # Normalize the normal
        normal /= np.linalg.norm(normal)

        return normal

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, width / float(height), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Handle mouse dragging for rotating the view."""
        # Left mouse button -> rotate around X and Y
        if buttons & pyglet.window.mouse.LEFT:
            self.ry += dx * 0.5  # Horizontal drag -> Y rotation
            self.rx -= dy * 0.5  # Vertical drag -> X rotation
        
        # Right mouse button -> rotate around Z
        if buttons & pyglet.window.mouse.RIGHT:
            self.rz += dx * 0.5  # Horizontal drag -> Z rotation

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Handle mouse scrolling for zooming in/out."""
        self.zoom += scroll_y * 5
