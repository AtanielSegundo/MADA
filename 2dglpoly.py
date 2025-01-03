import sys
import numpy as np
from scipy.spatial import Delaunay
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QWidget, QFileDialog, QMessageBox)
from PyQt5.QtOpenGL import QGLWidget
from OpenGL import GL


class DelaunayGLWidget(QGLWidget):
    def __init__(self):
        super(DelaunayGLWidget, self).__init__()
        self.z_coordinate = -3.0  # Initial z-coordinate for zoom
        self.vertices = None
        self.triangles = None

    def initializeGL(self):
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)  # Set background to white
        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        if height == 0:
            height = 1
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        # Adjust projection dynamically based on vertex range
        if self.vertices is not None:
            min_vals = self.vertices.min(axis=0)
            max_vals = self.vertices.max(axis=0)
            GL.glOrtho(min_vals[0] - 10, max_vals[0] + 10,
                       min_vals[1] - 10, max_vals[1] + 10,
                       -10.0, 10.0)
        else:
            GL.glOrtho(-1.5, 1.5, -1.5, 1.5, -10.0, 10.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, self.z_coordinate)  # Move back along z-axis

        if self.vertices is not None and self.triangles is not None:
            GL.glColor3f(0.5, 0.5, 0.5)  # Set triangle color
            for triangle in self.triangles:
                GL.glBegin(GL.GL_TRIANGLES)
                for vertex_index in triangle:
                    vertex = self.vertices[vertex_index]
                    GL.glVertex3f(vertex[0], vertex[1], self.z_coordinate)
                GL.glEnd()

    def update_vertices(self, vertices):
        """Update vertices and recompute triangulation."""
        self.vertices = vertices
        if self.vertices is not None:
            self.triangles = Delaunay(self.vertices).simplices
        else:
            self.triangles = None
        self.update()  # Trigger a repaint

    def wheelEvent(self, event):
        """Handle zoom in and zoom out with the mouse wheel."""
        zoom_factor = 0.1  # Adjust this value for faster/slower zoom
        delta = event.angleDelta().y()  # Get the scroll amount
        if delta > 0:
            self.z_coordinate += zoom_factor  # Zoom in
        else:
            self.z_coordinate -= zoom_factor  # Zoom out
        self.update()  # Trigger a repaint


class UpdateThread(QThread):
    request_update = pyqtSignal()

    def __init__(self):
        super(UpdateThread, self).__init__()
        self.running = True

    def run(self):
        while self.running:
            self.request_update.emit()
            self.msleep(1000)

    def stop(self):
        self.running = False


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Delaunay Triangulation with File Input")

        # OpenGL Widget
        self.gl_widget = DelaunayGLWidget()

        # Buttons and labels
        self.label = QLabel("Select a file with vertices to render:")
        self.file_button = QPushButton("Procurar Arquivo")
        self.file_button.clicked.connect(self.on_file_search)
        self.button = QPushButton("Click Me")
        self.button.clicked.connect(self.on_button_click)

        # Layout for the right panel
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label)
        right_layout.addWidget(self.file_button)
        right_layout.addWidget(self.button)
        right_layout.addStretch()

        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.gl_widget, stretch=3)
        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, stretch=1)

        # Update Thread
        self.update_thread = UpdateThread()
        self.update_thread.request_update.connect(self.gl_widget.update)
        self.update_thread.start()

    def on_file_search(self):
        """Handle file selection and vertex processing."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            vertices = self.process_file(file_path)
            if vertices is None:
                QMessageBox.warning(self, "Invalid File", "The selected file does not contain valid vertices.")
            else:
                self.gl_widget.update_vertices(vertices)
                self.label.setText(f"Loaded vertices from: {file_path}")

    def process_file(self, file_path):
        """Process the selected file and extract vertices."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                vertices = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        return None  # Invalid format
                    x, y = map(float, parts)
                    vertices.append([x, y])

                vertices = np.array(vertices, dtype=np.float32)
                return vertices
        except Exception as e:
            print(f"Error processing file: {e}")
            return None

    def on_button_click(self):
        self.label.setText("Button clicked!")

    def closeEvent(self, event):
        self.update_thread.stop()
        self.update_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
