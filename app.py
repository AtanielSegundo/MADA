import multiprocessing
import threading
import sys
import time
import random
import pyslm
import sys
import os
from math import ceil,floor
from PyQt5.QtWidgets import QApplication, QLineEdit,QMainWindow, QPushButton,QComboBox,\
                            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt,pyqtSignal, QObject
from PyQt5.QtGui import QIcon,QIntValidator,QDoubleValidator
from core.Layer import Layer
from core.Layer import SUPPORTED as layer_supports
from core.Part import SUPPORTED as part_supports
from core.visualize import showStl,ShowGeometrys
from core.TSP.strategy import AVAILABLE_END_TYPES,AVAILABLE_SOLVERS,AVAILABLE_GENERATORS,AVAILABLE_INITIAL_HEURISTICS
from core.TSP.strategy import Strategy
from core.visualize import SlicesPlotter
from matplotlib import pyplot as plt

ARR_AVAILABLE_END_TYPES = list(AVAILABLE_END_TYPES.keys())
ARR_AVAILABLE_SOLVERS = list(AVAILABLE_SOLVERS.keys())
ARR_AVAILABLE_GENERATORS = list(AVAILABLE_GENERATORS.keys())
ARR_AVAILABLE_INITIAL_HEURISTICS = list(AVAILABLE_INITIAL_HEURISTICS.keys())

get_first =  lambda _dict : next(iter(_dict.keys()))
GREY = "2E2E2E"
ACT_GREEN = "26932d"
ACT_ORANGE = "cc8c3c"
BACKGROUND_COLOR = GREY
BORDER_COLOR = "b0b0b0"
DEFAULT_COLOR = "4C4C4C"
IDENTIFIER_COLOR = "515151"
SELECTED_COLOR = "1f3f73"
WINDOW_TITLE_LOGO = "assets/png/pist_logo.png"
GLOBAL_APP_STYLE = f""" background-color: #{GREY}; 
                        color: #FFFFFF; 
                        font-size: 18px;
                    """
SELECTED = f"background-color: #{SELECTED_COLOR}; color: #FFFFFF;  border : 1px solid #{BORDER_COLOR};"
UNSELECTED = f"background-color: #{DEFAULT_COLOR}; color: #FFFFFF; border : 1px solid #{BORDER_COLOR};"
IDENTIFIER = f"background-color: #{IDENTIFIER_COLOR}; color: #FFFFFF; border: none;"
BLACKOUT   = f"background-color: rgba(0, 0, 0, 0.5); color: #AAAAAA; border: 1px solid #{BORDER_COLOR};"
SECTION_ID_SIZE = (400,60)
LAYER_MODE = 0
PART_MODE = 1

class RealTimePlot3DWithDisconnectedLines:
    def __init__(self,x_lim,y_lim,z_lim):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(*x_lim)
        self.ax.set_ylim(*y_lim)
        self.ax.set_zlim(*z_lim)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def add_line(self, x, y, z_index):
        """
        Adiciona uma linha conectando os pontos no plano z_index.
        """
        z = [z_index] * len(x)
        self.ax.plot(x, y, z, color='black', linewidth=1)
        plt.pause(0.01)


class AppState:
    def __init__(self):
        self.project_name = str(int(random.random()*1e5))
        self.target_file  = "assets/txt/formas/rabbit.txt"
        self.initial_heuristic = ARR_AVAILABLE_INITIAL_HEURISTICS[0]
        self.end_type   = ARR_AVAILABLE_END_TYPES[0]
        self.generator  = ARR_AVAILABLE_GENERATORS[0]
        self.tsp_solver = ARR_AVAILABLE_SOLVERS[0]
        self.multiple_processes_supported = False
        self.process_mode = LAYER_MODE
        # LAYER / PART SPECIFIC PARAMETERS
        self.scale  = 1.0       #float
        self.z_step = 7         #int
        self.seed   = 677       #int
        # STRATEGY SPECIFIC PARAMETERS
        self.n_clusters = 6     #int
        self.distance = 7       #float
        self.border_distance = 0    #float
        self.runs = 5   #int
        
def run_show(state:AppState):
    app = QApplication(sys.argv)
    if state.process_mode == LAYER_MODE:
        layer = Layer.From(state.target_file,scale=state.scale,z=state.z_step)
        layer.show()
    if state.process_mode == PART_MODE:
        showStl(state.target_file)  
    app.exec_()

def file_watcher(output_path,p_button,signal):
    """Monitors for .HARD_PROCESS_DONE file and emits a signal to update the GUI."""
    done_file = os.path.join(output_path, ".HARD_PROCESS_DONE")
    original_str = "Processando"
    cnt = 0
    while not os.path.exists(done_file):
        p_button.setText(original_str+"."*cnt)
        time.sleep(0.5)  
        cnt = (cnt + 1) % 4
    os.remove(done_file)  
    signal.emit()

def run_process(state:AppState):
    app = QApplication(sys.argv)
    out_path = f"./outputs/{state.project_name}/"
    if state.process_mode == LAYER_MODE:
        layer = Layer.From(state.target_file,scale=state.scale,z=state.z_step)
        _strategy =  Strategy(f"./outputs/{state.project_name}/",
                              n_clusters=state.n_clusters,
                              distance=state.distance,
                              border_distance=state.border_distance,
                              seed=state.seed,
                              runs=state.runs)
        grid,best_tour,metrics = _strategy.solve(layer,state.tsp_solver,state.generator,
                        end_type=state.end_type,
                        initial_heuristic=state.initial_heuristic)
        with open(out_path+"/.HARD_PROCESS_DONE","w") as f:
            pass
        
        plotter = SlicesPlotter([None], tile_direction='horizontal')
        plotter.set_random_usable_colors(state.n_clusters)
        plotter.set_background_colors(['black'])
        start_point = grid.points[best_tour.path[0]]
        end_point = grid.points[best_tour.path[-1]]
        plotter.draw_points([[start_point,end_point]],colors_maps=[[1,2]],markersize=3,edgesize=1)
        plotter.draw_vectors([grid.points],[best_tour.path],thick=1.25)
        plotter.draw_fig_title(metrics.tour_lenght.__ceil__())
        _fp = f"{os.path.basename(layer.tag)}_{state.generator}_{state.end_type}_{state.initial_heuristic}.png"
        plotter.save(os.path.join(out_path,_fp))
        plotter.show()
        
    if state.process_mode == PART_MODE:
        _part = pyslm.Part(os.path.basename(state.target_file).split(".")[0])
        _part.setGeometry(state.target_file)
        _part.scaleFactor = state.scale
        _part.dropToPlatform()
        x_min,y_min,z_min,x_max,y_max,z_max, = _part.boundingBox
        z_max = ceil(z_max)
        x_lim = [floor(x_min),ceil(x_max)]
        y_lim = [floor(y_min),ceil(y_max)]
        z_lim = [floor(z_min),z_max]
        plot = RealTimePlot3DWithDisconnectedLines(x_lim,y_lim,z_lim)
        total_slices_iter = z_max // state.z_step
        _strategy =  Strategy(f"./outputs/{state.project_name}/",
                              n_clusters=state.n_clusters,
                              distance=state.distance,
                              border_distance=state.border_distance,
                              seed=state.seed,
                              runs=state.runs)
        for ii in range(total_slices_iter):
            try:
                slice =  _part.getVectorSlice(state.z_step*ii)
                layer = Layer(slice,f"_slice_{ii}_",True)
                grid,best_tour,metrics = _strategy.solve(layer,state.tsp_solver,state.generator,end_type=state.end_type,initial_heuristic=state.initial_heuristic)
                grid_remaped = grid.points[best_tour.path]
                plot.add_line(grid_remaped[:,0],grid_remaped[:,1],state.z_step*ii)
            except:
                pass
        with open(out_path+"/.HARD_PROCESS_DONE","w") as f:
            pass
        plt.show(block=True)
    app.exec_()
    
class App(QMainWindow):
    process_completed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.identifier_widgets = {}
        self.setWindowTitle("PIST")  # Pixel Strategy
        self.setGeometry(100, 100, 1200, 720)
        self.setWindowIcon(QIcon(WINDOW_TITLE_LOGO))
        self.setStyleSheet(GLOBAL_APP_STYLE)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # PROJECT NAME SELECTOR 
        self.project_name_layout = QHBoxLayout()
        self.project_name_input = QLineEdit(self)
        self.project_name_input.setText(self.state.project_name)
        self.project_name_input.setStyleSheet(f"border: 1px solid #{BORDER_COLOR};")
        self.create_identifier("Projeto",self.project_name_layout,(200,40))
        self.project_name_input.setMinimumSize(600, 40)
        self.project_name_input.textChanged.connect(self.on_project_name_change)
        self.project_name_layout.addWidget(self.project_name_input)
        self.main_layout.addLayout(self.project_name_layout)

        ##self.search_button.setMaximumSize(150, 40)

        self.init_file_searcher()
        self.init_strategy_combos()
        self.init_process_parameters()

        # OPÇÕES DE PROCESSO
        self.create_identifier("Opções",self.main_layout,size=SECTION_ID_SIZE)
        
        self.process_options_container_layout = QHBoxLayout()
        self.process_options_container_layout.setSpacing(3)
        ##VIZUALIZER
        self.vizualize_button = QPushButton("Vizualizar", self)
        self.vizualize_button.setStyleSheet(f"background-color: #{ACT_GREEN}; color: #FFFFFF;padding: 10px;")
        self.vizualize_button.setMinimumSize(600,90)
        self.vizualize_button.clicked.connect(self.vizulize_target)
        self.process_options_container_layout.addWidget(self.vizualize_button)
        
        ##Processor
        self.processor_button = QPushButton("Processar", self)
        self.processor_button.setStyleSheet(f"background-color: #{ACT_GREEN}; color: #FFFFFF;padding: 10px;")
        self.processor_button.setMinimumSize(600,90)
        self.processor_button.clicked.connect(self.process_target)
        self.process_options_container_layout.addWidget(self.processor_button)
        
        self.main_layout.addLayout(self.process_options_container_layout)
        self.process_completed.connect(self.on_process_completed)

    def vizulize_target(self):
        p = multiprocessing.Process(target=run_show, args=(self.state,))
        p.start()

    def on_process_completed(self):
        self.processor_button.setStyleSheet(f"background-color: #{ACT_GREEN}; color: #FFFFFF;padding: 10px;")
        self.processor_button.setText("Processar")
        self.processor_button.setEnabled(True)

    def process_target(self):
        self.processor_button.setStyleSheet(f"background-color: #{ACT_ORANGE}; color: #FFFFFF;padding: 10px;")
        self.processor_button.setEnabled(False)
        p = multiprocessing.Process(target=run_process, args=(self.state,))
        p.start()
        out_path = f"./outputs/{self.state.project_name}/"
        watcher_thread = threading.Thread(target=file_watcher, args=(out_path,self.processor_button,self.process_completed))
        watcher_thread.daemon = True
        watcher_thread.start()
        
    def on_project_name_change(self,text:str):
        self.state.project_name = text
        
    def update_file_related_widgets(self):
        self.layer_mode_btn.setStyleSheet(UNSELECTED)
        self.part_mode_btn.setStyleSheet(UNSELECTED)
        if self.state.process_mode == LAYER_MODE:
            style = SELECTED
            if not self.state.multiple_processes_supported:
                self.part_mode_btn.setStyleSheet(BLACKOUT)
            self.layer_mode_btn.setStyleSheet(style)
        elif self.state.process_mode == PART_MODE:
            style = SELECTED
            if not self.state.multiple_processes_supported:
                self.layer_mode_btn.setStyleSheet(BLACKOUT)
            self.part_mode_btn.setStyleSheet(style)

    def create_identifier(self,name:str,layout,size=None):
        identifier = QPushButton(name, self)
        identifier.setStyleSheet(IDENTIFIER)  
        identifier.setEnabled(False)
        if size is not None:
            identifier.setMinimumSize(*size)
        self.identifier_widgets[name] = identifier
        layout.addWidget(identifier)

    def make_combo_box(self, identifier: str, combo_array, layout, parameter_name: str,
                   id_size=(60, 40), combo_size=(200, 40)):
        def updater(index):
            setattr(self.state, parameter_name, combo_box.currentText())

        container_layout = QHBoxLayout()
        self.create_identifier(identifier, container_layout, id_size)
        combo_box = QComboBox(self)
        combo_box.addItems(combo_array)
        combo_box.setMinimumSize(*combo_size)
        combo_box.setStyleSheet(f"border:1px solid #{BORDER_COLOR}")
        combo_box.setCurrentText(getattr(self.state, parameter_name))
        combo_box.currentIndexChanged.connect(updater)
        container_layout.addWidget(combo_box)
        layout.addLayout(container_layout)

    def make_parameter_box(self, identifier: str, parameter_name: str, validator, layout, type=int, length=3,
                        id_size=(200, 40), param_size=(200, 40)):
        def updater_builder(parameter_name, type):
            def updater(text):
                if text != '':
                    setattr(self.state, parameter_name, type(text))
            return updater

        container_layout = QHBoxLayout()
        paramater_box = QLineEdit()
        paramater_box.setText(str(getattr(self.state, parameter_name)))
        paramater_box.setMinimumSize(*param_size)
        paramater_box.setMaxLength(length)
        paramater_box.setValidator(validator)
        paramater_box.setAlignment(Qt.AlignCenter)
        paramater_box.textChanged.connect(updater_builder(parameter_name, type))
        self.create_identifier(identifier, container_layout, id_size)
        container_layout.addWidget(paramater_box)
        layout.addLayout(container_layout)    

    def init_process_parameters(self):
        prow1 = QHBoxLayout()
        self.make_parameter_box("Seed","seed",QIntValidator(),prow1)
        FMAX = 2**32-1
        prow2 = QHBoxLayout()
        float_validator = QDoubleValidator(0.00,FMAX,2)
        self.make_parameter_box("Escala","scale",float_validator,prow2,float)
        self.make_parameter_box("Espaçamento camadas","z_step",QIntValidator(),prow2)
        prow3 = QHBoxLayout()
        self.make_parameter_box("Clusters","n_clusters",QIntValidator(),prow3)
        self.make_parameter_box("Distância pontos","distance",QIntValidator(),prow3)
        prow4 = QHBoxLayout()
        self.make_parameter_box("Distância borda","border_distance",float_validator,prow4)
        self.make_parameter_box("Runs","runs",QIntValidator(),prow4)
        
        self.create_identifier("Parâmetros",self.main_layout,size=SECTION_ID_SIZE)
        self.main_layout.addLayout(prow1)
        self.main_layout.addLayout(prow2)
        self.main_layout.addLayout(prow3)
        self.main_layout.addLayout(prow4)

    def init_strategy_combos(self):
        self.first_strategy_combos_layout = QHBoxLayout()
        self.second_strategy_combos_layout = QHBoxLayout()
        self.make_combo_box("Algoritmo TSP",ARR_AVAILABLE_SOLVERS,
                            self.first_strategy_combos_layout,
                            "tsp_solver")
        self.make_combo_box("Gerar",ARR_AVAILABLE_GENERATORS,
                            self.first_strategy_combos_layout,
                            "generator")
        self.make_combo_box("Tipo de caminho",ARR_AVAILABLE_END_TYPES,
                            self.second_strategy_combos_layout,
                            "end_type")
        self.make_combo_box("Heuristica inicial",ARR_AVAILABLE_INITIAL_HEURISTICS,
                            self.second_strategy_combos_layout,
                            "initial_heuristic")
        self.create_identifier("Estratégia",self.main_layout,size=SECTION_ID_SIZE)
        self.main_layout.addLayout(self.first_strategy_combos_layout)
        self.main_layout.addLayout(self.second_strategy_combos_layout)

    def init_file_searcher(self):
        search_layout = QHBoxLayout()

        # SEARCH BUTTON
        self.search_button = QPushButton("Procurar modelo", self)
        self.search_button.setStyleSheet(f"background-color: #{DEFAULT_COLOR}; color: #FFFFFF; border: 1px solid #{BORDER_COLOR}; padding: 5px;")
        self.search_button.clicked.connect(self.open_file_dialog)
        self.search_button.setMaximumSize(150, 40)

        # FILE LABEL
        self.file_label = QLabel(self.state.target_file or "Nenhum arquivo selecionado", self)
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet(f"color: #FFFFFF; border: 1px solid #{BORDER_COLOR}; padding: 5px;")
        self.search_button.setMaximumSize(300, 40)
        
        # PROCESS MODE BUTTONS
        self.layer_mode_btn = QPushButton("fatia", self)
        self.layer_mode_btn.setMinimumSize(150, 40)
        self.layer_mode_btn.setStyleSheet(SELECTED)
        self.part_mode_btn = QPushButton("peça", self)
        self.part_mode_btn.setMinimumSize(150, 40)
        self.part_mode_btn.setStyleSheet(BLACKOUT)

        # Add widgets to main vertical layout
        search_layout.addWidget(self.file_label)  # Label at the top
        search_layout.addWidget(self.search_button)  # Search button below
        self.main_layout.addLayout(search_layout)
        self.create_identifier("Usar modelo como",self.main_layout,size=SECTION_ID_SIZE)
        #self.main_layout.addWidget(self.usage_button)  # Static button below

        # Horizontal layout for process buttons
        process_buttons_layout = QHBoxLayout()
        process_buttons_layout.addWidget(self.layer_mode_btn)
        process_buttons_layout.addWidget(self.part_mode_btn)

        # Add horizontal layout to the main layout
        self.main_layout.addLayout(process_buttons_layout)

        # Button click connections
        def update_mode_btn(app: App, mode):
            def mode_btns_updater():
                if app.state.multiple_processes_supported:
                    app.state.process_mode = mode
                    app.update_file_related_widgets()
            return mode_btns_updater

        self.layer_mode_btn.clicked.connect(update_mode_btn(self, LAYER_MODE))
        self.part_mode_btn.clicked.connect(update_mode_btn(self, PART_MODE))

    def is_file_supported(self, file_path: str) -> bool:
        file_extension = file_path.split(".")[-1]
        is_layer_supported = (file_extension in layer_supports)
        is_part_supported = (file_extension in part_supports)
        if is_layer_supported and is_part_supported:
            self.state.multiple_processes_supported = True
        elif is_layer_supported:
            self.state.multiple_processes_supported = False
            self.state.process_mode = LAYER_MODE
        elif is_part_supported:
            self.state.multiple_processes_supported = False
            self.state.process_mode = PART_MODE
        else:
            return False
        return True

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Arquivo", "", "Todos os Arquivos (*);;Arquivos de Texto (*.txt)", options=options)
        if file_path:
            if self.is_file_supported(file_path):
                self.state.target_file = file_path
                self.update_file_related_widgets()
                self.file_label.setText(file_path)
            else:
                QMessageBox.warning(self, "Arquivo Não Suportado", "O arquivo selecionado não é suportado em nenhum dos modos")
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())