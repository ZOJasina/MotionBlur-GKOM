import glfw
import numpy as np
import pywavefront
import logging
pywavefront.configure_logging(logging.ERROR)

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

def load_model_batched(path):
    """Reads whole model and arranges data in fromat [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z] """
    all_vertex_data = []
    draw_commands = []  # (start_index, vertex_count)
    current_vertex_index = 0

    try:
        scene = pywavefront.Wavefront(path, create_materials=True, parse=True)

        for mesh in scene.meshes.values():
            for material in mesh.materials:

                data = material.vertices
                vfs = material.vertex_format

                format_parts = vfs.split('_')
                offsets = {}
                current_offset = 0
                for part in format_parts:
                    offsets[part] = current_offset
                    if part == 'T2F': current_offset += 2
                    elif part == 'T3F': current_offset += 3
                    elif part == 'C3F': current_offset += 3
                    elif part == 'N3F': current_offset += 3
                    elif part == 'V3F': current_offset += 3

                vertex_format_size = current_offset

                pos_offset = offsets['V3F']
                norm_offset = offsets['N3F']

                vertex_count_for_this_material = 0

                for i in range(0, len(data), vertex_format_size):
                    all_vertex_data.append(data[i + pos_offset])     # px
                    all_vertex_data.append(data[i + pos_offset + 1]) # py
                    all_vertex_data.append(data[i + pos_offset + 2]) # pz

                    all_vertex_data.append(data[i + norm_offset])     # nx
                    all_vertex_data.append(data[i + norm_offset + 1]) # ny
                    all_vertex_data.append(data[i + norm_offset + 2]) # nz

                    vertex_count_for_this_material += 1

                draw_commands.append( (current_vertex_index, vertex_count_for_this_material) )
                current_vertex_index += vertex_count_for_this_material

        if not all_vertex_data:
            raise Exception("Error loading vertices")

        vertices_np = np.array(all_vertex_data, dtype=np.float32)

        return vertices_np, draw_commands

    except Exception as e:
        print(f"Error reading the model: {e}")
        glfw.terminate()
        return None, None