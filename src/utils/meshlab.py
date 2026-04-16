import pymeshlab


def meshlab_proc(meshpath):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(meshpath))
    ms.meshing_merge_close_vertices(threshold=pymeshlab.Percentage(1))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    # 填充孔洞
    # ms.meshing_repair_non_manifold_edges(method=0)
    # ms.meshing_repair_non_manifold_vertices(vertdispratio=0.05)
    # ms.meshing_close_holes(maxholesize=15)
    ms.save_current_mesh(
        str(meshpath),
        save_vertex_color=False,
        save_vertex_coord=False,
        save_face_color=False,
        save_wedge_texcoord=False,
    )
    ms.clear()
