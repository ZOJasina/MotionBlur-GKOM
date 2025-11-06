import glfw
import numpy as np
import pywavefront

def cube_vertices():
    return np.array([
        # positions         # normals
        # front face
        -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,    0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,

        # back face
        -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,    0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,

        # left face
        -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5,   -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,

        # right face
         0.5,  0.5,  0.5,    1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,    1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,    1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,    1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,    1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,    1.0,  0.0,  0.0,

        # bottom face
        -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,    0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,

        # top face
        -0.5,  0.5, -0.5,    0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,    0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,    0.0,  1.0,  0.0
    ], dtype=np.float32)

def triangle_vertices():
    return np.array([
        #  pos               normal
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.0,  0.5, 0.0,     0.0, 0.0, 1.0,
    ], dtype=np.float32)

def load_model(path):
    try:
        # scene = pywavefront.Wavefront("car.obj", create_materials=True, parse=True)
        scene = pywavefront.Wavefront(path, create_materials=True, parse=True)

        mesh = list(scene.meshes.values())[0]

        if not mesh.materials:
            raise Exception("cannot read vertices")

        material = mesh.materials[0]

        data = material.vertices
        vertex_format_string = material.vertex_format

        vertex_format_size = 0
        if "T2F" in vertex_format_string:
            vertex_format_size += 2
        if "T3F" in vertex_format_string:
            vertex_format_size += 3
        if "C3F" in vertex_format_string:
            vertex_format_size += 3
        if "N3F" in vertex_format_string:
            vertex_format_size += 3
        if "V3F" in vertex_format_string:
            vertex_format_size += 3

        if vertex_format_size == 0 or "V3F" not in vertex_format_string:
            raise Exception("Cannot find position data (V3F) in a model")

        positions = []
        for i in range(0, len(data), vertex_format_size):
            positions.append(data[i + vertex_format_size - 3])  # v_x
            positions.append(data[i + vertex_format_size - 2])  # v_y
            positions.append(data[i + vertex_format_size - 1])  # v_z

        vertices_np = np.array(positions, dtype=np.float32)
        vertex_count = len(positions) // 3

    except Exception as e:
        print(f"Error reading the model: {e}")
        glfw.terminate()
        return vertex_count, vertices_np