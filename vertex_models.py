import glfw
import numpy as np
import pywavefront
import logging
import os
pywavefront.configure_logging(logging.ERROR)


def parse_mtl_file(mtl_path, obj_dir):
    """Parse MTL file and return a dict mapping material names to texture paths and material properties."""
    material_data = {}
    if not os.path.exists(mtl_path):
        return material_data

    current_material = None
    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('newmtl'):
                    current_material = line.split()[1]
                    if current_material not in material_data:
                        material_data[current_material] = {
                            'texture_path': None,
                            'diffuse': [1.0, 1.0, 1.0],
                            'specular': [1.0, 1.0, 1.0],
                            'shininess': 32.0,
                            'ambient': [0.1, 0.1, 0.1],
                            'opacity': 1.0
                        }
                elif current_material:
                    if line.startswith('map_Kd'):
                        texture_path = line.split(' ', 1)[1].strip()
                        # Normalize path separators
                        texture_path = texture_path.replace('\\', os.sep)
                        # Make path relative to obj directory
                        if not os.path.isabs(texture_path):
                            full_path = os.path.join(obj_dir, texture_path)
                            # Try exact path first
                            if os.path.exists(full_path):
                                material_data[current_material]['texture_path'] = full_path
                            else:
                                # Try case-insensitive search in obj_dir
                                texture_name = os.path.basename(texture_path)
                                texture_dir = os.path.dirname(texture_path).lower()
                                found_path = None
                                for root, dirs, files in os.walk(obj_dir):
                                    # Check if we're in the right subdirectory (case-insensitive)
                                    rel_root = os.path.relpath(root, obj_dir)
                                    if rel_root == '.':
                                        rel_root = ''
                                    if rel_root.lower() == texture_dir or (texture_dir == '' and root == obj_dir):
                                        for file in files:
                                            if file.lower() == texture_name.lower():
                                                found_path = os.path.join(root, file)
                                                break
                                        if found_path:
                                            break
                                if found_path:
                                    material_data[current_material]['texture_path'] = found_path
                    elif line.startswith('Kd'):
                        parts = line.split()
                        if len(parts) >= 4:
                            material_data[current_material]['diffuse'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    elif line.startswith('Ks'):
                        parts = line.split()
                        if len(parts) >= 4:
                            material_data[current_material]['specular'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    elif line.startswith('Ka'):
                        parts = line.split()
                        if len(parts) >= 4:
                            material_data[current_material]['ambient'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    elif line.startswith('Ns'):
                        parts = line.split()
                        if len(parts) >= 2:
                            material_data[current_material]['shininess'] = float(parts[1])
                    elif line.startswith('d'):
                        parts = line.split()
                        if len(parts) >= 2:
                            material_data[current_material]['opacity'] = float(parts[1])
    except Exception as e:
        print(f"Error parsing MTL file {mtl_path}: {e}")

    return material_data


def load_model_batched(path):
    """Reads whole model and arranges data in format [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, tex_u, tex_v]
    Returns: (vertices_np, draw_commands, texture_paths, material_properties)
    where texture_paths is a list of texture file paths for each material
    and material_properties is a list of dicts with diffuse, specular, shininess, ambient for each material"""
    all_vertex_data = []
    draw_commands = []  # (start_index, vertex_count)
    texture_paths = []  # texture file path for each material
    material_properties = []  # material properties for each material
    current_vertex_index = 0

    try:
        obj_dir = os.path.dirname(path)

        # Parse MTL file if it exists
        mtl_data = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('mtllib'):
                        mtl_filename = line.split(' ', 1)[1].strip()
                        mtl_path = os.path.join(obj_dir, mtl_filename)
                        # Try exact filename first, then try with _obj suffix
                        if not os.path.exists(mtl_path):
                            base_name = os.path.splitext(mtl_filename)[0]
                            alt_mtl_path = os.path.join(obj_dir, base_name + '_obj.mtl')
                            if os.path.exists(alt_mtl_path):
                                mtl_path = alt_mtl_path
                        mtl_data = parse_mtl_file(mtl_path, obj_dir)
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

                # Get texture path and material properties from MTL data
                texture_path = None
                material_name = getattr(material, 'name', None)
                mat_props = {
                    'diffuse': [1.0, 1.0, 1.0],
                    'specular': [1.0, 1.0, 1.0],
                    'shininess': 32.0,
                    'ambient': [0.1, 0.1, 0.1],
                    'opacity': 1.0
                }

                if material_name and material_name in mtl_data:
                    mtl_info = mtl_data[material_name]
                    texture_path = mtl_info.get('texture_path')
                    mat_props['diffuse'] = mtl_info.get('diffuse', [1.0, 1.0, 1.0])
                    mat_props['specular'] = mtl_info.get('specular', [1.0, 1.0, 1.0])
                    mat_props['shininess'] = mtl_info.get('shininess', 32.0)
                    mat_props['ambient'] = mtl_info.get('ambient', [0.1, 0.1, 0.1])
                    mat_props['opacity'] = mtl_info.get('opacity', 1.0)

                # Verify texture exists
                if texture_path and not os.path.exists(texture_path):
                    texture_path = None

                texture_paths.append(texture_path)
                material_properties.append(mat_props)

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

        return vertices_np, draw_commands, texture_paths, material_properties

    except Exception as e:
        print(f"Error reading the model: {e}")
        glfw.terminate()
        return None, None, None, None
