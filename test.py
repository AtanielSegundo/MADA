available_opts = []
desired_function = ""

HELP_FUNCTION_NAMES = ["help", "-h", "--h"]


class CTX:
    def __init__(self):
        pass


ctx = CTX()


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
            setattr(ctx_locals, var_name, eval(var_val))
    return ctx_locals


def test_grid(*params):
    from core.geometry import fill_geometrys_with_points
    from core.clipper import offsetSVG, offsetTXT
    from core.visualize import SlicesPlotter
    ctx.distance = 3
    ctx.iter = 2
    ctx.hole = 60
    update_ctx_with_params(ctx, params)
    if any([(param in HELP_FUNCTION_NAMES) for param in params]):
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
    from test_h.clipper import offsetBridge, offsetRedStuff
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
    if any([(param in HELP_FUNCTION_NAMES) for param in params]):
        show_opts_help()


def test_offset_txt(*params):
    from core.geometry import nthgone
    from core.clipper import offsetTXT
    import os
    import numpy as np

    def create_two_semi_circles(ray: float, separation: float, c_points=1000):
        semi_circle_1 = nthgone(c_points, ray=ray)[(c_points):]
        semi_circle_2 = nthgone(c_points, ray=ray)[0:(c_points)]
        semi_circle_1 = np.array(
            [[semi_circle_1[idx % len(semi_circle_1)]] for idx in range(len(semi_circle_1)+1)])
        semi_circle_2 = np.array(
            [[semi_circle_2[idx % len(semi_circle_2)]] for idx in range(len(semi_circle_2)+1)])
        return [semi_circle_1, semi_circle_2]
    ctx.iteracoes = 40
    ctx.offset = -2
    ctx.path = "assets/txt/formas"
    ctx.precisao = 1e3
    update_ctx_with_params(ctx, params)
    if any([(param in HELP_FUNCTION_NAMES) for param in params]):
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

    if any([(param in HELP_FUNCTION_NAMES) for param in params]):
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
    from core.Grid import generatePointsAndClusters
    from core.visualize import SlicesPlotter
    from core.geometry import generate_square_box_by_lenght,getPolygonsCenter
    import os
    ctx.distance = 5
    ctx.cluster_n = 6
    ctx.output = f"outputs/d_{ctx.distance}_cn_{ctx.cluster_n}"
    if check_params("ttf"):
        from core.text import str2Polygons
        ctx.text = "WAAM\nMADA"
        ctx.font = "assets\\ttf\\arial.ttf"
        ctx.scale = 60
        update_ctx_with_params(ctx, params)
        ctx.output = f"outputs/d_{ctx.distance}_cn_{ctx.cluster_n}"
        if any([(param in HELP_FUNCTION_NAMES) for param in params]):
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
        if any([(param in HELP_FUNCTION_NAMES) for param in params]):
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
        if any([(param in HELP_FUNCTION_NAMES) for param in params]):
            show_opts_help(ctx)
            return
        else:
            for arquivo in os.listdir(ctx.path):
                print(f"Lendo {arquivo}")
                if "truss" not in arquivo:
                    forma = getSliceStl(os.path.join(ctx.path, arquivo), z=1)
                else:
                    forma = getSliceStl(os.path.join(ctx.path, arquivo), z=1, scaleFactor=0.25)
                file_name = arquivo.replace(".stl", "_klusters.png")
                grid, pred, centers = generatePointsAndClusters(forma,  distance=ctx.distance, clusters_n=ctx.cluster_n, fliped_y=True)
                _plt = SlicesPlotter([forma])
                _plt.set_random_usable_colors(ctx.cluster_n)
                _plt.draw_points([grid],colors_maps=[pred]).draw_points([centers],colors_maps=[list(range(0,ctx.cluster_n))],edgesize=3).save(os.path.join(ctx.output,file_name))
                print()
    if any([(param in HELP_FUNCTION_NAMES) for param in params]):
        show_opts_help()

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
