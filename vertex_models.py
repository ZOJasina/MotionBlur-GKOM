import glfw
import numpy as np
import pywavefront
import logging
import os
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

def parse_mtl_file(mtl_path, obj_dir):
    """Parse MTL file and return a dict mapping material names to texture paths."""
    material_textures = {}
    if not os.path.exists(mtl_path):
        return material_textures

    current_material = None
    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('newmtl'):
                    current_material = line.split()[1]
                elif line.startswith('map_Kd') and current_material:
                    texture_path = line.split(' ', 1)[1].strip()
                    # Make path relative to obj directory
                    if not os.path.isabs(texture_path):
                        full_path = os.path.join(obj_dir, texture_path)
                        if os.path.exists(full_path):
                            material_textures[current_material] = full_path
    except Exception as e:
        print(f"Error parsing MTL file {mtl_path}: {e}")

    return material_textures

def load_model_batched(path):
    """Reads whole model and arranges data in format [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, tex_u, tex_v]
    Returns: (vertices_np, draw_commands, texture_paths)
    where texture_paths is a list of texture file paths for each material, relative to the obj file directory"""
    all_vertex_data = []
    draw_commands = []  # (start_index, vertex_count)
    texture_paths = []  # texture file path for each material
    current_vertex_index = 0

    try:
        obj_dir = os.path.dirname(path)

        # Parse MTL file if it exists
        mtl_textures = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('mtllib'):
                        mtl_filename = line.split(' ', 1)[1].strip()
                        mtl_path = os.path.join(obj_dir, mtl_filename)
                        mtl_textures = parse_mtl_file(mtl_path, obj_dir)
                        break
        except Exception as e:
            print(f"Warning: Could not parse MTL file: {e}")

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
                has_tex = 'T2F' in offsets
                tex_offset = offsets.get('T2F', 0)

                # Get texture path from material
                texture_path = None
                material_name = getattr(material, 'name', None)

                if material_name and material_name in mtl_textures:
                    texture_path = mtl_textures[material_name]

                # alternative to manual but not working
                # if not texture_path and material_name and hasattr(scene, 'materials') and material_name in scene.materials:
                #     mtl = scene.materials[material_name]
                #     if hasattr(mtl, 'map_Kd') and mtl.map_Kd:
                #         texture_path = mtl.map_Kd
                #         # Make path relative to obj directory
                #         if not os.path.isabs(texture_path):
                #             texture_path = os.path.join(obj_dir, texture_path)
                #     elif hasattr(mtl, 'texture') and mtl.texture:
                #         texture_path = getattr(mtl.texture, 'path', None) or str(mtl.texture)
                #         if texture_path and not os.path.isabs(texture_path):
                #             texture_path = os.path.join(obj_dir, texture_path)

                # Verify texture exists
                if texture_path and not os.path.exists(texture_path):
                    texture_path = None

                texture_paths.append(texture_path)

                vertex_count_for_this_material = 0

                for i in range(0, len(data), vertex_format_size):
                    all_vertex_data.append(data[i + pos_offset])     # px
                    all_vertex_data.append(data[i + pos_offset + 1]) # py
                    all_vertex_data.append(data[i + pos_offset + 2]) # pz

                    all_vertex_data.append(data[i + norm_offset])     # nx
                    all_vertex_data.append(data[i + norm_offset + 1]) # ny
                    all_vertex_data.append(data[i + norm_offset + 2]) # nz

                    # Add texture coordinates if available, otherwise use (0, 0)
                    if has_tex:
                        all_vertex_data.append(data[i + tex_offset])     # u
                        all_vertex_data.append(data[i + tex_offset + 1]) # v
                    else:
                        all_vertex_data.append(0.0)  # u
                        all_vertex_data.append(0.0)  # v

                    vertex_count_for_this_material += 1

                draw_commands.append( (current_vertex_index, vertex_count_for_this_material) )
                current_vertex_index += vertex_count_for_this_material

        if not all_vertex_data:
            raise Exception("Error loading vertices")

        vertices_np = np.array(all_vertex_data, dtype=np.float32)

        return vertices_np, draw_commands, texture_paths

    except Exception as e:
        print(f"Error reading the model: {e}")
        glfw.terminate()
        return None, None, None