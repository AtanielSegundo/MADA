import numpy as np
import pyclipr.pyclipr
from core.clipper import offsetPaths
from core.geometry import rotate_180,flip_horizontal,nthgone
from core.visualize import ShowGeometrys
    
def generateBridge(dim_size:float) -> np.ndarray :
    square = nthgone(4,dim_size) 
    circle = nthgone(100,dim_size/2,center_p=(0,3*dim_size/4)) 
    pc = pyclipr.Clipper()
    pc.scaleFactor = int(1000)
    pc.addPaths([square],pyclipr.Subject)
    pc.addPath(circle,pyclipr.Clip)
    result = pc.execute(pyclipr.Difference, pyclipr.FillRule.EvenOdd)[0]
    return rotate_180(np.array([result[i%len(result)] for i in range(len(result)+1)]))
    
def offsetBridge(bridge_size:float = 4,DISTANCE=-1,ITER=3,show=False) :
    bridge   = generateBridge(bridge_size)
    offseted = offsetPaths([bridge],DISTANCE,ITER)
    if show : ShowGeometrys([[bridge],offseted],spliter=2)
    return bridge,offseted

def getRedStuffContours() :
    import cv2
    image = cv2.imread('assets/png/red_stuff.png')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)  
    upper_red1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)  
    upper_red2 = np.array([255, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_list = []
    for contour in contours:
        if len(contour) > 2:  
            contour_list.append(np.squeeze(contour, axis=1)) 
    contour_arr = np.array([contour_list[i%len(contour_list)] for i in range(len(contour_list)+1)])[0]
    return flip_horizontal(rotate_180(contour_arr))

def offsetRedStuff(DISTANCE=-40,ITER=1,show=False) :
    redStuff = getRedStuffContours()
    offseted = offsetPaths([redStuff],DISTANCE,ITER)
    if show: ShowGeometrys([offseted],spliter=1)
    return redStuff,offseted

available_opts = []
desired_function = ""

HELP_FUNCTION_NAMES = ["help", "-h", "--h"]

class CTX:
    def __init__(self):
        pass


ctx = CTX()

help_asked = lambda params : any(param in HELP_FUNCTION_NAMES for param in params) 


def check_params(*opts):
    result = all([(n in params or "all" in params) for n in opts])
    if result == False:
        available_opts.append([*opts])
    return result


def show_opts_help(ctx_params: CTX = None):
    print(f"OPCOES VALIDAS PARA O TESTE {desired_function}")
    for opt in available_opts:
        print(" ".join(opt))
    if ctx_params is not None:
        print(f"PARAMETROS VALIDOS PARA O TESTE {desired_function}")
        for key,val in ctx_params.__dict__.items():
            if "__" not in key:
                print(f"{key}={repr(val)}")

def update_ctx_with_params(ctx_locals, params):
    for param in params:
        if "=" in param:
            var_name, var_val = param.split("=")
            try:
                val = eval(var_val)
            except:
                val = var_val
            setattr(ctx_locals, var_name, val)
    return ctx_locals


def test_grid(*params):
    from core.geometry import fill_geometrys_with_points
    from core.clipper import offsetSVG, offsetTXT
    from core.visualize import SlicesPlotter
    ctx.distance = 3
    ctx.iter = 2
    ctx.hole = 60
    update_ctx_with_params(ctx, params)
    if help_asked(params):
        show_opts_help(ctx)
    else:
        rabbit, hole, _ = offsetSVG("assets/svg/rabbit.svg", DISTANCE=-
                                    ctx.distance, ITER=ctx.iter, HOLE_RAY=ctx.hole, scale=1)
        layer = [rabbit, hole]
        grid = fill_geometrys_with_points(layer, ctx.distance)
        SlicesPlotter([layer], split_in=1).draw_points([grid]).show()
        layer, _ = offsetTXT("assets\\txt\\formas\\teste_biela.txt", iter=ctx.iter, offset=-ctx.distance)
        grid = fill_geometrys_with_points(layer, ctx.distance)
        SlicesPlotter([layer], split_in=1).draw_points([grid]).show()


def test_clipper(*params):
    import numpy as np
    from core.clipper import offsetSVG
    from core.transform import fold_3d_array_to_2d_using_NaN_separator
    if check_params("rabbit", "hole"):
        offsetSVG("assets/rabbit.svg", HOLE_RAY=0)
        rabbit, hole, offsets = offsetSVG(
            "assets/rabbit.svg", HOLE_RAY=40, scale=1, show=True)
        fig = np.concatenate(
            (rabbit, np.array([[np.nan, np.nan]]), hole), axis=0)
        np.savetxt("assets/rabbit_hole.txt", fig, delimiter=",")
        np.savetxt("assets/offsets_hole.txt",
                   fold_3d_array_to_2d_using_NaN_separator(offsets), delimiter=",")
    if check_params("rabbit"):
        rabbit, hole, offsets = offsetSVG(
            "assets/rabbit.svg", HOLE_RAY=0, scale=1, show=True)
        np.savetxt("assets/rabbit.txt", rabbit, delimiter=",")
        np.savetxt("assets/offsets.txt",
                   fold_3d_array_to_2d_using_NaN_separator(offsets), delimiter=",")
    if check_params("bridge"):
        bridge, offseted = offsetBridge(DISTANCE=-1, ITER=3, show=True)
        np.savetxt("assets/bridge.txt", bridge, delimiter=",")
        np.savetxt("assets/bridge_offset.txt",
                   fold_3d_array_to_2d_using_NaN_separator(offseted), delimiter=",")
    if check_params("red"):
        redstuff, offseted = offsetRedStuff(DISTANCE=-40, ITER=1, show=True)
        np.savetxt("assets/global_loop.txt", redstuff, delimiter=",")
        np.savetxt("assets/global_loop_offset.txt",
                   fold_3d_array_to_2d_using_NaN_separator(offseted), delimiter=",")
    if help_asked(params):
        show_opts_help()


def test_offset_txt(*params):
    from core.geometry import nthgone
    from core.clipper import offsetTXT
    import os
    import numpy as np
    ctx.iteracoes = 40
    ctx.offset = -2
    ctx.path = "assets/txt/formas"
    ctx.precisao = 1e3
    update_ctx_with_params(ctx, params)
    if help_asked(params):
        show_opts_help(ctx)
    else:
        for arquivo in os.listdir(ctx.path):
            print(f"lendo {arquivo}")
            offsetTXT(os.path.join(ctx.path, arquivo), ctx.iteracoes,
                      ctx.offset, precisao=ctx.precisao, show=True)


def test_ttf(*params):
    import numpy as np
    from core.visualize import ShowGeometrys
    from core.clipper import offsetPaths
    from core.geometry import generate_square_box_by_lenght,getPolygonsCenter,online_mean
    from core.text import str2Polygons
    ctx.test_font = "assets\\ttf\\arial.ttf"
    ctx.text = "WAAM"
    ctx.size = 32
    ctx.distance = 20
    ctx.iter = 2
    update_ctx_with_params(ctx, params)

    if help_asked(params):
        show_opts_help(ctx)
    else:
        waam_p, offsetx, offsety = str2Polygons(
            ctx.text, ctx.test_font, font_size=ctx.size, _return_offsets=True)
        square = generate_square_box_by_lenght(
            max(abs(offsetx), abs(offsety)), getPolygonsCenter(waam_p))
        assemble = waam_p+[square]
        offsets = offsetPaths(assemble, -ctx.distance, ctx.iter)
        ShowGeometrys([assemble, offsets], spliter=2)


def test_kmeans(*params):
    from core.Points.Grid import generatePointsAndClusters
    from core.visualize import SlicesPlotter
    from core.geometry import generate_square_box_by_lenght,getPolygonsCenter
    import os
    ctx.distance = 5
    ctx.cluster_n = 6
    ctx.output = f"outputs/d_{ctx.distance}_cn_{ctx.cluster_n}"
    os.makedirs(ctx.output,exist_ok=True)
    if check_params("ttf"):
        from core.text import str2Polygons
        ctx.text = "WAAM\nMADA"
        ctx.font = "assets\\ttf\\arial.ttf"
        ctx.scale = 60
        update_ctx_with_params(ctx, params)
        ctx.output = f"outputs/d_{ctx.distance}_cn_{ctx.cluster_n}"
        if help_asked(params):
            show_opts_help(ctx)
            return
        else:
            waam_p, offsetx, offsety = str2Polygons(ctx.text,ctx.font,ctx.scale,_return_offsets=True)
            square = generate_square_box_by_lenght(max(abs(offsetx), abs(offsety)), getPolygonsCenter(waam_p))
            forma = [square]
            forma.extend(waam_p)
            grid, pred, centers = generatePointsAndClusters(forma, distance=ctx.distance, clusters_n=ctx.cluster_n)
            _plt = SlicesPlotter([None]).set_background_colors(["black"])
            _plt.set_random_usable_colors(ctx.cluster_n)
            _plt.draw_points([grid],colors_maps=[pred]).draw_points([centers],colors_maps=[list(range(0,ctx.cluster_n))],edgesize=3).save(os.path.join(ctx.output,"ttf.png"))
    if check_params("txt"):
        from core.transform import geometrys_from_txt_nan_separeted
        ctx.path = "assets/txt/formas"
        update_ctx_with_params(ctx, params)
        ctx.output = f"outputs/d_{ctx.distance}_cn_{ctx.cluster_n}"
        if help_asked(params):
            show_opts_help(ctx)
            return
        else:
            for arquivo in os.listdir(ctx.path):
                print(f"Lendo {arquivo}")
                forma = geometrys_from_txt_nan_separeted(os.path.join(ctx.path, arquivo))
                file_name = arquivo.replace(".txt", "_klusters.png")
                grid, pred, centers = generatePointsAndClusters(
                    forma, distance=ctx.distance, clusters_n=ctx.cluster_n)
                _plt = SlicesPlotter([forma])
                _plt.set_random_usable_colors(ctx.cluster_n)
                _plt.draw_points([grid],colors_maps=[pred]).draw_points([centers],colors_maps=[list(range(0,ctx.cluster_n))],edgesize=3).save(os.path.join(ctx.output,file_name))
    if check_params("stl"):
        from core.slicing import getSliceStl
        ctx.path = "assets/3d"
        update_ctx_with_params(ctx, params)
        ctx.output = f"outputs/d_{ctx.distance}_cn_{ctx.cluster_n}"
        if help_asked(params):
            show_opts_help(ctx)
            return
        else:
            for arquivo in os.listdir(ctx.path):
                print(f"Lendo {arquivo}")
                if "truss" not in arquivo:
                    forma = getSliceStl(os.path.join(ctx.path, arquivo), z=1)
                else:
                    forma = getSliceStl(os.path.join(ctx.path, arquivo), z=1, scale=0.25)
                file_name = arquivo.replace(".stl", "_klusters.png")
                grid, pred, centers = generatePointsAndClusters(forma,  distance=ctx.distance, clusters_n=ctx.cluster_n, fliped_y=True)
                _plt = SlicesPlotter([None])
                _plt.set_random_usable_colors(ctx.cluster_n)
                _plt.draw_points([grid],colors_maps=[pred]).draw_points([centers],colors_maps=[list(range(0,ctx.cluster_n))],edgesize=3).save(os.path.join(ctx.output,file_name))
                print()
    if help_asked(params):
        show_opts_help()


def test_slm(*params):
    from core.visualize import showStl 
    from core.slicing import sliceStlVector
    ctx.path = ""
    ctx.z_step = 1
    ctx.n_slices = 100
    ctx.scale = 1.0 
    ctx.show = False
    model_params = {
        "custom": {"path": "assets/3d/flange16furos.stl", "z_step": 1, "n_slices": 100, "scale": 1.0},
        "bonnie": {"path": "assets/3d/bonnie.stl", "z_step": 2, "n_slices": 200, "scale": 1.0},
        "truss":  {"path": "assets/3d/truss_.stl", "z_step": 1, "n_slices": 100, "scale": 1.0},
        "Foice":  {"path": "assets/3d/Petro_foice_c.stl", "z_step": 1, "n_slices": 100, "scale": 1.0},
        "Flange": {"path": "assets/3d/flange16furos.stl", "z_step": 2.5, "n_slices": 20, "scale": 1.0},
        "frame_guide": {"path": "assets/3d/frameGuide.stl", "z_step": 4, "n_slices": 10, "scale": 1.0},
    }
    if help_asked(params):
        show_opts_help(ctx)
        return
    for model, m_params in model_params.items():
        if check_params(model):
            ctx.path = m_params["path"]
            ctx.z_step = m_params["z_step"]
            ctx.n_slices = m_params["n_slices"]
            ctx.scale = m_params["scale"]
            update_ctx_with_params(ctx, params)
            if ctx.show:
                showStl(ctx.path)
            sliceStlVector(ctx.path, n_slices=ctx.n_slices, z_step=ctx.z_step, scaleFactor=ctx.scale)
            return
    if help_asked(params):
        show_opts_help(ctx)

def test_tsp(*params):
    ctx.path = "assets/txt/formas/rabbit.txt"
    ctx.distance = 10
    ctx.cluster_n = 2
    ctx.iterations = 20
    update_ctx_with_params(ctx, params)
    if help_asked(params):
        show_opts_help(ctx)
        return
    else:
        pass

def test_path_gen(*params):
    from core.Layer import Layer
    from core.geometry import fill_geometrys_with_points
    from core.visualize import SlicesPlotter
    from core.Tour import generateDummyTour,generateCH,generateCH_with_dummy
    ctx.file = "assets/svg/rabbit.svg"
    ctx.seed = 777
    ctx.z = 10
    ctx.scale = 0.5
    ctx.distance = 5
    ctx.borders = 0
    ctx.end_point = 2
    update_ctx_with_params(ctx, params)
    if help_asked(params):
        show_opts_help(ctx)
        return
    else:
        layer = Layer.From(ctx.file, scale=ctx.scale, z=ctx.z)
        forma = layer.data
        _plt = SlicesPlotter([None,None,None],tile_direction="horizontal")
        _plt.set_random_usable_colors(6)
        _plt.set_background_colors(['black','black','black'])
        grid = fill_geometrys_with_points(forma,delta=ctx.distance,figure_sep=ctx.borders,fliped_y=layer.is_y_flipped)
        path_dummy = generateDummyTour(0,ctx.end_point, len(forma[0])+1)
        path_ch_dummy = generateCH_with_dummy(grid, 0,ctx.end_point)
        path_ch = generateCH(grid)
        _plt.draw_vectors([grid,grid,grid],[path_ch,path_dummy[:-1],path_ch_dummy[:-1]])
        _plt.draw_points([[grid[path_ch[0]], grid[path_ch[-1]]],
                      [grid[path_dummy[:-1][0]], grid[path_dummy[:-1][-1]]],
                      [grid[path_ch_dummy[:-1][0]], grid[path_ch_dummy[:-1][-1]]]],
                     colors_maps=[[0,1],[0,1],[0,1]])
        _plt.show()

def test_off_slices(*params):
    from core.slicing import sliceStlVector
    from core.clipper import offsetPaths
    ctx.n_slices = 20
    ctx.iterations = 50
    ctx.z_step = 1 
    ctx.off_dist = None
    ctx.mode = "3d"
    ctx.path = ""
    ctx.scale = 1.0
    update_ctx_with_params(ctx, params)
    stlSlicer = lambda path,mode,**kwargs : sliceStlVector(path,d2_mode=(True if mode=="2d" else False),**kwargs)
    if ctx.off_dist is None :
        offseter = None
    else : 
        offseter = lambda gArr : offsetPaths(gArr,-ctx.off_dist,ctx.iterations)
    if check_params("custom"):
        stlSlicer(ctx.path,ctx.mode,n_slices=ctx.n_slices,z_step=ctx.z_step,scaleFactor=ctx.scale,offset_fn=offseter)
    if check_params("petro"):
        ctx.path = "assets/3d/Petro_foice.stl"
        stlSlicer(ctx.path,ctx.mode,n_slices=ctx.n_slices,z_step=ctx.z_step,scaleFactor=ctx.scale,offset_fn=offseter)
    if check_params("truss"):
        ctx.path = "assets/3d/truss_.stl"
        stlSlicer(ctx.path,ctx.mode,n_slices=ctx.n_slices,z_step=ctx.z_step,scaleFactor=ctx.scale,offset_fn=offseter)
    if help_asked(params):
        show_opts_help(ctx)
        return


if __name__ == "__main__":
    from sys import argv
    desired_function, *params = argv[1:]
    namespaced_fn = f"test_{desired_function}"
    available_fns = [fn for fn in globals().keys() if "test" in fn]
    if namespaced_fn in available_fns:
        globals().get(namespaced_fn)(*params)
    elif desired_function in HELP_FUNCTION_NAMES:
        print("TESTES DISPONIVEIS")
        _ = [print(t.split("test_")[-1]) for t in available_fns]