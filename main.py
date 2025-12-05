import glfw
from OpenGL.GL import *
import glm
import ctypes
from car import Car
from PIL import Image
import numpy as np
import os
from object3d import Object3D
from vertex_models import load_model_batched


def load_shader(shader_file, shader_type):
    """reads and compiles shader from file"""
    try:
        with open(shader_file, "r") as f:
            shader_source = f.read()
    except FileNotFoundError:
        print(f"Error: file not found {shader_file}")
        return None

    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8")
        print(f"Error compiling shader ({shader_file}):\n{log}")
        return None

    return shader


def create_shader_program(vertex_file, fragment_file):
    """Creates shader program"""
    vertex_shader = load_shader(vertex_file, GL_VERTEX_SHADER)
    fragment_shader = load_shader(fragment_file, GL_FRAGMENT_SHADER)

    if not vertex_shader or not fragment_shader:
        return None

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8")
        print(f"Error linking program:\n{log}")
        return None

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def create_default_texture():
    """Creates a default white 1x1 texture."""
    texture_id = glGenTextures(1)
    if texture_id == 0:
        print("Failed to generate default texture ID")
        return None

    glBindTexture(GL_TEXTURE_2D, texture_id)

    white_pixel = np.array([255, 255, 255, 255], dtype=np.uint8)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, white_pixel)

    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id


def load_texture(texture_path):
    """Loads a texture from file and returns OpenGL texture ID."""
    if not texture_path or not os.path.exists(texture_path):
        print(f"Texture path does not exist: {texture_path}")
        return None

    try:
        img = Image.open(texture_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img_data = np.array(img, dtype=np.uint8)
        width, height = img.size

        img_data = np.flipud(img_data)

        texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        error = glGetError()
        if error != GL_NO_ERROR:
            print(
                f"OpenGL error after loading texture {texture_path}: {error}")
            glDeleteTextures(1, [texture_id])
            glBindTexture(GL_TEXTURE_2D, 0)
            return None

        glBindTexture(GL_TEXTURE_2D, 0)
        print(
            f"Successfully loaded texture: {texture_path} (ID: {texture_id})")
        return texture_id

    except Exception as e:
        print(f"Error loading texture {texture_path}: {e}")
        return None


def setup_model(vertices_np):
    """Creates VAO + VBO for given vertex array.
    Expected format: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, tex_u, tex_v] (8 floats per vertex)"""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_STATIC_DRAW)

    stride = 8 * vertices_np.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, vbo


def render_model(shader, vao, vertex_count, model, view, projection):
    """Renders a model with given transformation matrices."""
    glUseProgram(shader)
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))

    # Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix3fv(glGetUniformLocation(shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)
    glBindVertexArray(0)
    return

def cache_uniforms(program):
    """Call once after program creation. Returns a dict of frequently used uniform locations."""
    keys = [
        "model", "view", "projection", "normalMatrix",
        "texture_diffuse1", "useTexture",
        "material.diffuse", "material.specular", "material.shininess", "material.ambient", "material.opacity",
        "u_alpha", "viewPos", "cameraMode"
    ]
    locs = {}
    for k in keys:
        locs[k] = glGetUniformLocation(program, k)
    return locs


def render_complex_model(shader, vao, draw_commands, model, view, projection, texture_ids=None, default_texture=None, material_properties=None, locs=None):
    """Renders a complex (batched) model using draw commands"""
    glUseProgram(shader)

    # Set uniforms
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))
    if locs["model"] != -1:
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    if locs["view"] != -1:
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    if locs["projection"] != -1:
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    if locs["normalMatrix"] != -1:
        glUniformMatrix3fv(glGetUniformLocation(shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

    glBindVertexArray(vao)

    glActiveTexture(GL_TEXTURE0)
    texture_loc = glGetUniformLocation(shader, "texture_diffuse1")
    use_texture_loc = glGetUniformLocation(shader, "useTexture")

    if texture_loc != -1:
        glUniform1i(texture_loc, 0)

    for idx, (start_index, vertex_count) in enumerate(draw_commands):
        # Apply material properties for this material if available
        if material_properties and idx < len(material_properties):
            mat_props = material_properties[idx]
            if locs["material.diffuse"] != -1:
                glUniform3f(glGetUniformLocation(shader, "material.diffuse"), mat_props['diffuse'][0], mat_props['diffuse'][1], mat_props['diffuse'][2])
            if locs["material.specular"] != -1:
                glUniform3f(glGetUniformLocation(shader, "material.specular"), mat_props['specular'][0], mat_props['specular'][1], mat_props['specular'][2])
            if locs["material.shininess"] != -1:
                glUniform1f(glGetUniformLocation(shader, "material.shininess"), mat_props['shininess'])
            if locs["material.ambient"] != -1:
                glUniform3f(glGetUniformLocation(shader, "material.ambient"), mat_props['ambient'][0], mat_props['ambient'][1], mat_props['ambient'][2])
            if locs["material.opacity"] != -1:
                glUniform1f(glGetUniformLocation(shader, "material.opacity"), mat_props['opacity'])


        tex_id = None
        if texture_ids and idx < len(texture_ids) and texture_ids[idx] is not None:
            try:
                tex_id = int(texture_ids[idx])
            except (ValueError, TypeError):
                tex_id = None

        if tex_id is not None and tex_id > 0:
            glBindTexture(GL_TEXTURE_2D, tex_id)
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 1)
        else:
            if default_texture is not None and default_texture > 0:
                glBindTexture(GL_TEXTURE_2D, default_texture)
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 0)

        glDrawArrays(GL_TRIANGLES, start_index, vertex_count)

    glBindTexture(GL_TEXTURE_2D, 0)
    glBindVertexArray(0)
    return


def load_model(path):
    """Loads a model and returns Object3D with vao, vbo, draw_commands, texture_ids, and material_properties"""
    vertices_np, draw_commands, texture_paths, material_properties = load_model_batched(path)
    if vertices_np is None:
        print("Failed to load model, terminating.")
        glfw.terminate()
        return None

    vao, vbo = setup_model(vertices_np)

    # Load textures
    texture_ids = []
    for idx, tex_path in enumerate(texture_paths):
        if tex_path:
            tex_id = load_texture(tex_path)
            texture_ids.append(tex_id)
            if tex_id is None:
                print(
                    f"Warning: Failed to load texture {idx} for model {path}: {tex_path}")
        else:
            texture_ids.append(None)

    loaded_count = sum(1 for tid in texture_ids if tid is not None and tid > 0)
    return Object3D(vao, vbo, draw_commands, texture_ids, material_properties)


def initialize_window():
    if not glfw.init():
        print("Cannot initiate GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Motion Blur", None, None)
    if not window:
        glfw.terminate()
        print("Cannot create a window")
        return

    glfw.make_context_current(window)

    shader_program = create_shader_program("phong.vert", "phong.frag")
    glEnable(GL_DEPTH_TEST)
    if not shader_program:
        glfw.terminate()
        return
    return window, shader_program


def check_collision(car, collision_center_pos, collision_radius):
    min_distance = 100
    for c in car.get_collision_centers():
        distance = glm.length(c - collision_center_pos)
        if distance < min_distance:
            min_distance = distance

    min_allowed_distance = car.collision_radius + collision_radius

    return min_distance < min_allowed_distance, min_distance, min_allowed_distance


def main():
    window, shader_program = initialize_window()
    if not window:
        return

    def uni(name):
        loc = glGetUniformLocation(shader_program, name)
        if loc == -1:
            print(f"Warning: uniform '{name}' not found (location = -1). Maybe it's optimized out or name mismatch.")
        return loc

    default_texture = create_default_texture()
    locs = cache_uniforms(shader_program)

    # === LOAD MODELS ===
    car3d = load_model("objects/porsche_obj.obj")
    steering_wheel = (load_model("objects/steering_wheel.obj")
                      .set_material(specular=[0.1, 0.1, 0.1], shininess=8.0)
                      .translate(-0.15, -0.24, -0.16).rotate(0, -90, 0).scale(0.05))

    static_objects = []

    grass_plane = (load_model("objects/grass.obj")
                   .translate(1.0, -0.55, -1.0).scale(40))
    static_objects.append(grass_plane)
    racetrack = (load_model("objects/racetrack.obj")
                 .translate(0.5, -0.5, 0.0).scale(0.5))
    static_objects.append(racetrack)
    starting_line = (load_model("objects/start.obj")
                     .translate(0.505, -0.49, -2.25).scale(0.5)
                     .add_collider((-2.0, 0, 0), 0.1).add_collider((1.0, 0, 0), 0.1))
    static_objects.append(starting_line)
    finish_line = (load_model("objects/finish.obj")
                   .translate(0.505, -0.49, 3.25).scale(0.5)
                   .add_collider((-2.0, 0, -2.4), 0.1).add_collider((1.0, 0, -2.4), 0.1))
    static_objects.append(finish_line)

    for z in range(7, -15, -1):
        for x in range(0, 2):
            static_objects.append(load_model("objects/pine_tree.obj")
                                  .set_material(specular=[0.2, 0.2, 0.2], shininess=16.0)
                                  .translate(-2.2-x*1.6, -0.5, z).scale(0.23)
                                  .add_collider((0, 0, 0), 0.2))
    for z in range(0, 6, 2):
        static_objects.append(load_model("objects/pine_tree.obj")
                                .set_material(specular=[0.2, 0.2, 0.2], shininess=16.0)
                                .translate(27, -0.5, z+5).scale(0.15)
                                .add_collider((0, 0, 0), 0.2))

    for z in range(2, -5, -1):
        for x in range(0, 2):
            static_objects.append(load_model("objects/green_tree.obj")
                                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                                  .translate(3.2+x*3, -0.5, z*2).scale(0.2)
                                  .add_collider((0, 0, 0), 0.1))
    for z in range(0, 6, 1):
        static_objects.append(load_model("objects/pole.obj")
                                .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                                .translate(15, -0.5, z-20).scale(0.2)
                                .add_collider((0, 0, 0), 0.1))
        static_objects.append(load_model("objects/pole.obj")
                                .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                                .translate(3, -0.5, z-17).scale(0.2)
                                .add_collider((0, 0, 0), 0.1))
    big_bushes = []
    for z in range(0, 2, 1):
        big_bush = (load_model("objects/big_bush.obj")
                            .set_material(specular=[0.3, 0.3, 0.3], shininess=16.0)
                            .translate(10, -0.5, -30+z*57).scale(0.8).rotate(0,90,0))
        for x in range(30, -24, -2):
            big_bush.add_collider((x, 0, 0), 0.95)
        static_objects.append(big_bush)
        big_bushes.append(big_bush)

    for x in range(0, 2, 1):
        big_bush_side = (load_model("objects/big_bush.obj")
                            .set_material(specular=[0.3, 0.3, 0.3], shininess=16.0)
                            .translate(35-x*45, -0.5, 0).scale(0.8))
        for z in range(24, -29, -2):
            big_bush_side.add_collider((0, 0, z), 0.95)
        static_objects.append(big_bush_side)
        big_bushes.append(big_bush_side)

    fallen_green_tree = (load_model("objects/green_tree.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(20.0, -0.46, 0.0).scale(0.5).rotate(0, 0, 90)
                  .add_collider((-0.15, 0, 0), 0.15)
                  .add_collider((-0.45, 0, 0), 0.15)
                  .add_collider((-0.75, 0, 0), 0.15)
                  .add_collider((-1, 0, 0), 0.2)
                  .add_collider((-1.4, 0, 0), 0.2)
                  .add_collider((-1.8, 0, 0), 0.2)
                  .add_collider((-2.6, 0, 0), 0.5)
                  .add_collider((-2.6, 0, 0.5), 0.5)
                  .add_collider((-3.4, 0, -0.4), 0.6))
    static_objects.append(fallen_green_tree)
    big_green_tree = (load_model("objects/green_tree.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(22.0, -0.5, 6.0).scale(0.45).rotate(0, 90, 0)
                  .add_collider((0, 0, 0), 0.3))
    static_objects.append(big_green_tree)

    savanna_tree = (load_model("objects/savanna_tree.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(16.0, -0.5, 10.0).scale(0.3).rotate(0, 0, 0)
                  .add_collider((0, 0, 0), 0.2))
    static_objects.append(savanna_tree)
    savanna_tree3 = (load_model("objects/savanna_tree.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(21.0, -1.3, 10.0).scale(0.3).rotate(0, -75, 0)
                  .add_collider((0, 0.8, 0), 0.3))
    static_objects.append(savanna_tree3)
    savanna_tree2 = (load_model("objects/savanna_tree.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(13.0, -0.5, -15.0).scale(0.5).rotate(0, 45, 0)
                  .add_collider((0, 0, 0), 0.3))
    static_objects.append(savanna_tree2)

    race_barricade_red = (load_model("objects/race_barricade_red.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(20.0, -0.5, -20.0).scale(0.3).rotate(0, 0, 0)
                  .add_collider((0, 0, 0.2), 0.2)
                  .add_collider((0, 0, -0.2), 0.2))
    static_objects.append(race_barricade_red)
    race_barricade_white = (load_model("objects/race_barricade_white.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(20.0, -0.5, -18.0).scale(0.3).rotate(0, -5, 0)
                  .add_collider((0, 0, 0.2), 0.2)
                  .add_collider((0, 0, -0.2), 0.2))
    static_objects.append(race_barricade_white)
    race_barricade_red2 = (load_model("objects/race_barricade_red.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(20.0, -0.5, -16.0).scale(0.3).rotate(0, -10, 0)
                  .add_collider((0, 0, 0.2), 0.2)
                  .add_collider((0, 0, -0.2), 0.2))
    static_objects.append(race_barricade_red2)

    pole = (load_model("objects/pole.obj")
                  .set_material(specular=[0.3, 0.3, 0.3], shininess=32.0)
                  .translate(18.0, -0.4, -5.0).scale(0.2).rotate(0, -10, 0)
                  .add_collider((0, 0, 0), 0.1))
    static_objects.append(pole)


    # === CAMERA, LIGHT ===
    glClearColor(0.55, 0.70, 0.95, 1.0)

    glUseProgram(shader_program)
    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)
    camera_mode = 0  # 0: 3rd POV, 1: 1st POV

    def key_callback(window, key, scancode, action, mods):
        nonlocal camera_mode
        if key == glfw.KEY_C and action == glfw.PRESS:
            camera_mode = 1 - camera_mode
            glUniform1i(uni("cameraMode"), camera_mode)
    glfw.set_key_callback(window, key_callback)

    glUniform3f(uni("lightPos"), 5.0, 10.0, -5.0)
    glUniform3f(uni("viewPos"), 0.0, 0.0, 2.0)

    glUniform3f(uni("light.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(uni("light.diffuse"), 0.8, 0.8, 0.8)
    glUniform3f(uni("light.specular"), 1.0, 1.0, 1.0)

    # === CAR ===
    car = Car(initial_position=glm.vec3(-0.1, -0.35, 0.0))
    previous_time = glfw.get_time()
    car.prev_position = glm.vec3(car.position)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glUniform1f(uni("u_alpha"), 1.0)
    glUniform1i(uni("cameraMode"), camera_mode)

    # Main loop
    while not glfw.window_should_close(window):
        # Handle window resize
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)
        projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        time_val = glfw.get_time()
        delta_time = time_val - previous_time
        previous_time = time_val
        car.update(window, delta_time)

        # === CAMERA ===
        up = glm.vec3(0.0, 1.0, 0.0)
        forward_x, forward_z = -glm.sin(car.angle), -glm.cos(car.angle)
        forward_vector = glm.vec3(forward_x, 0.0, forward_z)

        if camera_mode == 0:
            backward_vector = -forward_vector
            distance_behind, height_above, look_ahead = 1.75, 0.75, 1.0
            eye = car.position + (backward_vector * distance_behind) + glm.vec3(0.0, height_above, 0.0)
            center = car.position + (forward_vector * look_ahead)
        else:
            downward_tilt = glm.vec3(0.0, -0.20, 0.0)
            right_vector = glm.vec3(glm.cos(car.angle), 0.0, -glm.sin(car.angle))
            height_offset, right_offset, forward_offset, look_ahead = 0.214, -0.045, -0.067, 1.0
            eye = car.position + glm.vec3(0.0, height_offset, 0.0) + (right_vector * right_offset) + (forward_vector * forward_offset)
            center = eye + (forward_vector * look_ahead) + downward_tilt

        view = glm.lookAt(eye, center, up)
        if locs["viewPos"] != -1:
            glUniform3f(locs["viewPos"], eye.x, eye.y, eye.z)

        # === STATIC SCENERY ===
        for obj in static_objects:
            obj_distance = glm.length(car.position - obj.get_position())
            to_point = obj.get_position() - car.position
            dot = glm.dot(car.direction, to_point)
            if obj not in big_bushes:
                if obj.colliders and dot < 0 and obj_distance > 2: # point is behind
                    continue
            if obj.material_properties:
                render_complex_model(shader_program, obj.vao, obj.draw_commands, obj.get_trans_matrix(), view, projection, obj.texture_ids, default_texture, obj.material_properties, locs)
            else:
                obj.prepare_material(shader_program)
                render_complex_model(shader_program, obj.vao, obj.draw_commands, obj.get_trans_matrix(), view, projection, obj.texture_ids, default_texture, None, locs)

        # === COLLISIONS ===
        collided = False
        for obj in static_objects:
            if not obj.colliders:
                continue  # skip objects without collision

            for center, radius in obj.colliders.items():
                center_pos = glm.vec3(obj.translation[0], obj.translation[1], obj.translation[2]) + center
                collided, dist, min_dist = check_collision(car, center_pos, radius)
                if collided:
                    # Push the car out of the tree
                    push_dir = car.position - center_pos
                    push_dir = glm.normalize(push_dir)
                    push_dir.y = 0

                    correction = push_dir * (min_dist - dist)
                    car.position += correction
                    # Stop the car or reduce speed
                    car.velocity = glm.vec3(0, 0, 0)
                    car.speed = 0.0
                    break
            if collided:
                break

        # === MOTION BLUR ===
        if not collided:
            smoothing_factor = 0.2  # 0 = no smoothing, 1 = full lag
            car.smoothed_velocity = glm.mix(car.smoothed_velocity, car.position - car.prev_position, smoothing_factor)
            velocity = car.smoothed_velocity
            speed = glm.length(velocity)

            blur_steps = 10

            base_blur_strength = 700
            max_speed = 5.0
            max_dist = 7

            motion_blur_factor = base_blur_strength * glm.smoothstep(0.0, max_speed, speed)

            for i in range(blur_steps):
                alpha = 1.0 / blur_steps
                step_fraction = (i + 1) / blur_steps

                for obj in static_objects:
                    obj_distance = glm.length(car.position - obj.get_position())
                    if obj_distance > max_dist:
                        continue
                    to_point = obj.get_position() - car.position
                    dot = glm.dot(car.direction, to_point)
                    if dot < 0: # point is behind
                        continue

                    obj_distance = max(obj_distance, 0.001)
                    distance_factor = 1.0 / obj_distance

                    obj_offset = velocity * step_fraction * motion_blur_factor * distance_factor
                    view_obj_blur = glm.translate(view, -obj_offset)
                    glUniform1f(uni("u_alpha"), alpha)
                    if obj.material_properties:
                        render_complex_model(shader_program, obj.vao, obj.draw_commands, obj.get_trans_matrix(), view_obj_blur, projection, obj.texture_ids, default_texture, obj.material_properties, locs)
                    else:
                        obj.prepare_material(shader_program)
                        render_complex_model(shader_program, obj.vao, obj.draw_commands, obj.get_trans_matrix(), view_obj_blur, projection, obj.texture_ids, default_texture, None, locs)
            glUniform1f(uni("u_alpha"), 1.0)

        # === STEERING WHEEL ===
        wheel_local_scale = glm.scale(glm.mat4(1.0), glm.vec3(0.3))
        wheel_local_base_rotate = glm.rotate(glm.mat4(1.0), glm.radians(90.0), glm.vec3(0,1,0))

        wheel_local_translate = glm.translate(glm.mat4(1.0), glm.vec3(0.25, 0.5, 0.8))   # local offset in car space
        wheel_local_transform = wheel_local_translate * (wheel_local_base_rotate) * wheel_local_scale

        # compose with car model
        car_trans_matrix = car.get_trans_matrix()
        wheel_trans_matrix = car_trans_matrix * wheel_local_transform
        if steering_wheel.material_properties:
            render_complex_model(shader_program, steering_wheel.vao, steering_wheel.draw_commands, wheel_trans_matrix, view, projection, steering_wheel.texture_ids, default_texture, steering_wheel.material_properties, locs)
        else:
            steering_wheel.prepare_material(shader_program)
            render_complex_model(shader_program, steering_wheel.vao, steering_wheel.draw_commands, wheel_trans_matrix, view, projection, steering_wheel.texture_ids, default_texture, None, locs)
        # === CAR === (rendered last due to windowpane opacity)
        render_complex_model(shader_program, car3d.vao, car3d.draw_commands, car_trans_matrix, view, projection, car3d.texture_ids, default_texture, car3d.material_properties, locs)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()
