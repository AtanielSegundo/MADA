import sys
from multiprocessing import Process, Pipe
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSplitter, QWidget, QVBoxLayout
from PyQt5.QtGui import QWindow


def run_opengl_widget(pipe):
    """Child process for the OpenGL widget."""
    from PyQt5.QtOpenGL import QGLWidget
    from PyQt5.QtCore import QTimer
    from OpenGL import GL

    class GLWidget(QGLWidget):
        def __init__(self):
            super(GLWidget, self).__init__()
            self.angle = 0  # Initial rotation angle

            # Timer to update the cube rotation
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_rotation)
            self.timer.start(16)  # ~60 FPS (16 ms interval)

        def initializeGL(self):
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)

def paintGL(self):
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glLoadIdentity()
    GL.glTranslatef(0.0, 0.0, -5.0)
    GL.glRotatef(self.angle, 1.0, 1.0, 0.0)

    # Draw cube faces
    GL.glBegin(GL.GL_QUADS)

    # Front face
    GL.glColor3f(1.0, 0.0, 0.0)  # Red
    GL.glVertex3f(-1.0, -1.0, 1.0)
    GL.glVertex3f(1.0, -1.0, 1.0)
    GL.glVertex3f(1.0, 1.0, 1.0)
    GL.glVertex3f(-1.0, 1.0, 1.0)

    # Back face
    GL.glColor3f(0.0, 1.0, 0.0)  # Green
    GL.glVertex3f(-1.0, -1.0, -1.0)
    GL.glVertex3f(-1.0, 1.0, -1.0)
    GL.glVertex3f(1.0, 1.0, -1.0)
    GL.glVertex3f(1.0, -1.0, -1.0)

    # Left face
    GL.glColor3f(0.0, 0.0, 1.0)  # Blue
    GL.glVertex3f(-1.0, -1.0, -1.0)
    GL.glVertex3f(-1.0, -1.0, 1.0)
    GL.glVertex3f(-1.0, 1.0, 1.0)
    GL.glVertex3f(-1.0, 1.0, -1.0)

    # Right face
    GL.glColor3f(1.0, 1.0, 0.0)  # Yellow
    GL.glVertex3f(1.0, -1.0, -1.0)
    GL.glVertex3f(1.0, 1.0, -1.0)
    GL.glVertex3f(1.0, 1.0, 1.0)
    GL.glVertex3f(1.0, -1.0, 1.0)

    # Top face
    GL.glColor3f(1.0, 0.0, 1.0)  # Magenta
    GL.glVertex3f(-1.0, 1.0, -1.0)
    GL.glVertex3f(-1.0, 1.0, 1.0)
    GL.glVertex3f(1.0, 1.0, 1.0)
    GL.glVertex3f(1.0, 1.0, -1.0)

    # Bottom face
    GL.glColor3f(0.0, 1.0, 1.0)  # Cyan
    GL.glVertex3f(-1.0, -1.0, -1.0)
    GL.glVertex3f(1.0, -1.0, -1.0)
    GL.glVertex3f(1.0, -1.0, 1.0)
    GL.glVertex3f(-1.0, -1.0, 1.0)

    GL.glEnd()

    # Draw cube edges with red or white lines
    GL.glColor3f(1.0, 1.0, 1.0)  # White lines for edges
    GL.glBegin(GL.LINES)
    edges = [
        (-1, -1, -1), (-1, -1, 1), (-1, -1, 1), (-1, 1, 1),
        (-1, 1, 1), (-1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, -1), (1, -1, 1), (1, -1, 1), (1, 1, 1),
        (1, 1, 1), (1, 1, -1), (1, 1, -1), (1, -1, -1),
        (-1, -1, -1), (1, -1, -1), (-1, -1, 1), (1, -1, 1),
        (-1, 1, 1), (1, 1, 1), (-1, 1, -1), (1, 1, -1),
    ]
    for edge in edges:
        GL.glVertex3f(*edge)
    GL.glEnd()

        def update_rotation(self):
            self.angle += 2  # Increment rotation angle
            if self.angle >= 360:
                self.angle -= 360
            self.update()  # Trigger repaint

    app = QApplication(sys.argv)
    gl_window = GLWidget()
    gl_window.show()
    pipe.send(int(gl_window.winId()))
    sys.exit(app.exec_())


def run_other_widget(pipe):
    from PyQt5.QtWidgets import QApplication, QLabel

    app = QApplication(sys.argv)
    label = QLabel("Other Widget\nHandled by a Separate Process")
    label.setAlignment(Qt.AlignCenter)
    label.resize(400, 400)
    label.show()
    pipe.send(int(label.winId()))  # Send window ID to parent process
    sys.exit(app.exec_())


class Window(QWidget):
    def __init__(self, opengl_id, other_id):
        super(Window, self).__init__()

        splitter = QSplitter(self)

        # Embed other widget
        other_window = QWindow.fromWinId(other_id)
        other_widget = QWidget.createWindowContainer(other_window)
        splitter.addWidget(other_widget)

        # Embed OpenGL widget
        opengl_window = QWindow.fromWinId(opengl_id)
        opengl_widget = QWidget.createWindowContainer(opengl_window)
        splitter.addWidget(opengl_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)


if __name__ == '__main__':
    opengl_parent_conn, opengl_child_conn = Pipe()
    other_parent_conn, other_child_conn = Pipe()

    opengl_process = Process(target=run_opengl_widget, args=(opengl_child_conn,))
    other_process = Process(target=run_other_widget, args=(other_child_conn,))

    opengl_process.start()
    other_process.start()

    opengl_id = opengl_parent_conn.recv()
    other_id = other_parent_conn.recv()

    app = QApplication(sys.argv)
    window = Window(opengl_id, other_id)
    window.resize(800, 600)
    window.show()

    try:
        sys.exit(app.exec_())
    finally:
        if opengl_process.is_alive():
            opengl_process.terminate()
        if other_process.is_alive():
            other_process.terminate()
