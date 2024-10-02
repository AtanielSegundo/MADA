from matplotlib import pyplot
from core.geometry import ShowGeometrys 
from os.path import basename 
from commons.clipperutils import offsetPaths
import numpy as np
import pyslm
import pyglet
from pyglet.gl import *
from typing import *
from scipy.spatial import Delaunay

def showStl(stl_path:str) :
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    solidPart.geometry.show()

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

def showSlices3d(slices:List[List[np.ndarray]]):
    """
    Use Pyglet to visualize the 3D slices in a 3D space.

    Parameters:
    - slices: A list of lists of 3D numpy arrays, where each numpy array represents
              a set of points in the 3D space.
    """
    window = SliceViewer(slices)
    pyglet.app.run()


def showSlices3d_matplot(slices: List[List[np.ndarray]], fig_title: str = ""):
    """
    Visualize the 3D slices in a single 3D plot.

    Parameters:
    - slices: List of lists of 3D numpy arrays. Each sublist corresponds to a slice, 
              and each numpy array within the sublist corresponds to a part of that slice.
    - fig_title: Title of the figure (default is '3D Slices Visualization')
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for slice_3d_list in slices:
        for slice_3d in slice_3d_list:
            ax.plot(slice_3d[:, 0], slice_3d[:, 1], slice_3d[:, 2],linestyle='-')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    pyplot.title(fig_title)
    pyplot.show()

def sliceStlVector(stl_path: str, n_slices=6, z_step=14,
                   origin:List[float]=[5.0, 10.0, 0.0],
                   rotation:np.ndarray[float]=np.array([0, 0, 30]),
                   scaleFactor:float=2.0,
                   dropToPlataform:bool=True,
                   d2_mode=False,use_matplot=True):
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    #Transform the part: Rotate, translate, scale, and drop to platform
    solidPart.scaleFactor = scaleFactor
    solidPart.origin   = origin
    solidPart.rotation = rotation
    if dropToPlataform : solidPart.dropToPlatform() 
    slices = []
    for i in range(n_slices):
        slice_2d_list = solidPart.getVectorSlice((i + 1) * z_step)
        # offsetPaths(slice_2d_list,-5,20)
        if d2_mode :
            slices.append(slice_2d_list)
        else :
            slice_3d_list = []
            for slice_2d in slice_2d_list:
                z_values = np.full(slice_2d.shape[0], (i + 1) * z_step)  
                slice_3d = np.column_stack((slice_2d, z_values))         
                slice_3d_list.append(slice_3d)
            slices.append(slice_3d_list)  
    if d2_mode: ShowGeometrys(slices)    
    else: showSlices3d_matplot(slices) if use_matplot else showSlices3d(slices)

def sliceStlRaster(stl_path: str, n_slices=6, z_step=14,
                   origin: List[float] = [5.0, 10.0, 0.0],
                   rotation: np.ndarray = np.array([0, 0, 30]),
                   scaleFactor: float = 2.0,
                   dropToPlatform: bool = True,
                   spliter=3):
    solidPart = pyslm.Part(basename(stl_path).split(".")[0])
    solidPart.setGeometry(stl_path)
    solidPart.scaleFactor = scaleFactor
    solidPart.origin = origin
    solidPart.rotation = rotation
    if dropToPlatform:
        solidPart.dropToPlatform()

    ##Unity for rasterization is mm/px and dpi is px/inch
    dpi = 300.0
    resolution = 25.4 / dpi  
    rows = n_slices//spliter + n_slices%spliter 
    #Create subplots for the number of slices
    _, axs = pyplot.subplots(rows,spliter)
    
    if n_slices == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one slice
    
    for idx in range(n_slices):
        _slice = solidPart.getTrimeshSlice((idx + 1) * z_step)
        if _slice:
            sliceImage = _slice.rasterize(pitch=resolution, origin=solidPart.boundingBox[:2])
            axs[idx//spliter,idx%spliter].imshow(np.array(sliceImage), cmap='gray', origin='lower')
            axs[idx//spliter,idx%spliter].set_title(f"Slice {idx + 1}")
        else:
            axs[idx//spliter,idx%spliter].text(0.5, 0.5, "No Slice Data", ha='center', va='center')
            axs[idx//spliter,idx%spliter].axis('off')  
    pyplot.show()



if __name__ == '__main__' :
    # showStl("assets/3d/bonnie.stl")
    # sliceStlVector('assets/3d/bonnie.stl',z_step=2,n_slices=200,scaleFactor=1.0)
    
    # showStl("assets/3d/bonnie.stl")
    # sliceStlVector('assets/3d/bonnie.stl',n_slices=200,z_step=2,scaleFactor=1)
    # sliceStlVector('assets/3d/bonnie.stl',n_slices=20,z_step=20,scaleFactor=1)

    # showStl("assets/3d/frameGuide.stl")
    # sliceStlVector('assets/3d/frameGuide.stl',n_slices=100,z_step=1,scaleFactor=1)
    sliceStlVector('assets/3d/frameGuide.stl',n_slices=10,z_step=4,scaleFactor=1)