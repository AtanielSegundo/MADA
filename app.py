import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

def init_opengl():
    """
    Initialize OpenGL context and set up the projection.
    """
    glClearColor(0.2, 0.2, 0.2, 1.0)  # Grey background for better visibility
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D rendering
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800 / 600, 0.1, 50.0)  # Perspective projection
    glMatrixMode(GL_MODELVIEW)

def draw_vector(start, end, color=(1.0, 0.0, 0.0)):
    """
    Draw a single vector as a line from 'start' to 'end' in a given 'color'.
    """
    glColor3f(*color)  # Set the color of the line
    glBegin(GL_LINES)
    glVertex3f(*start)  # Start point of the vector
    glVertex3f(*end)    # End point of the vector
    glEnd()

def debug_output():
    """
    Debugging function to ensure OpenGL is working and objects are being drawn.
    """
    print("OpenGL Version:", glGetString(GL_VERSION).decode())
    print("Renderer:", glGetString(GL_RENDERER).decode())
    print("Vendor:", glGetString(GL_VENDOR).decode())
    print("GLSL Version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

def main():
    """
    Main application loop.
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Draw Vectors with PyOpenGL")

    init_opengl()
    debug_output()  # Output OpenGL version and renderer information

    camera_position = [0, 0, -5]  # Position the camera backward for visibility
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen with background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(*camera_position)  # Apply camera transformation

        # Draw test vectors
        draw_vector((0, 0, 0), (1, 1, 1), (1.0, 0.0, 0.0))  # Red vector
        draw_vector((0, 0, 0), (-1, 1, 1), (0.0, 1.0, 0.0))  # Green vector
        draw_vector((0, 0, 0), (0, -1, 1), (0.0, 0.0, 1.0))  # Blue vector

        pygame.display.flip()  # Swap buffers to display the rendered frame
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
